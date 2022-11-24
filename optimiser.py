import geopandas as gpd
import pandas as pd
import numpy as np
import gurobipy as gp
import sys

# this stuff exists for local testing, so it can be run independent of the notebook for now
d = pd.read_excel("./data/Demand_data.xlsx")
p = pd.read_excel("./data/Interest_points.xlsx")
d['geometry'] = gpd.GeoSeries.from_wkt(d['geometry'])
gd = gpd.GeoDataFrame(d, geometry='geometry', crs='EPSG:4326')

n_periods = 4
n_regions = len(d["Ref"])

# certain adjustable parameters
max_chargers_per_region = 1000
max_supply_per_region = 1000000
time_limit = 60
MIP_gap = 0.08
POI_importance = 1


def get_period_names(upto_n: int):
    if upto_n > 4:
        print("N too high, must be integer <= 4")
        return None
    return ["Demand_{}".format(p) for p in range(upto_n)]


def get_neighbors(region, data):
    # we need this to access the neighbors of any given region, \
    # converts from string rep. of a list "[0,1,2]" to an actual list [0,1,2]
    return [int(x) for x in data['NEIGHBORS'][region].strip('][').split(', ')]


def get_new_dataframe(demand_data: pd.DataFrame, interest_points: pd.DataFrame):

    m = gp.Model()

    # maximum rates for each type of charger in kWh/year. assuming constant maximum usage for now.
    charging_capacity = {'Slow': 3500, 'Fast': 5200, 'Rapid': 50500}

    existing_chargers = np.array([demand_data["Number of {} Charging Points".format(
        speed)] for speed in charging_capacity.keys()])

    # demand[i][j] is the demand (in kWh/year) on year i of region j
    demand = np.array([demand_data[period]
                       for period in get_period_names(n_periods)])

    # supply[i][j] is same as demand, but represents supply
    supply = m.addMVar(demand.shape)

    # indexing chargers at region:
    # chargers_at_region['fast'][0][2] to get the number of 'fast' chargers during time period 0 in region 3
    chargers_added_at_region = {
        p: m.addMVar(demand.shape, vtype=gp.GRB.INTEGER, lb=0) for p in charging_capacity.keys()}

    for speed in charging_capacity.keys():
        for i in range(n_periods-1):
            for j in range(n_regions):
                # enforcing continuity, in that the number of chargers can only increase year-on-year.
                # without this constraint, our configuration can look completely different between years
                m.addConstr(
                    chargers_added_at_region[speed][i+1][j] >= chargers_added_at_region[speed][i][j])

    # simple sum over every type of charger over every region for each time period
    number_of_chargers_year = m.addMVar(
        (n_periods, len(charging_capacity.keys())))

    # a ['slow', 'fast', 'rapid'] shape for each region, indicating how much energy
    # is supplied by a specific source to each region

    capacities = [m.addMVar((len(charging_capacity.keys()), n_regions))
                  for _ in range(n_periods)]\

    # indicating the amount of energy supplied BY a region on each year (from chargers inside that region)
    # index capacity_from_region[i][j] : capacity from region j on year i
    capacity_from_region = [m.addMVar((n_regions))
                            for _ in range(n_periods)]

    # indicating how much energy can potentially be used by a region's neighbors
    # index capacity_from_region[i][j][k] : energy from region j used by region k on year i
    capacity_to_neighbors = [m.addMVar((n_regions, n_regions))
                             for _ in range(n_periods)]

    # indicating how much of a region's produced energy is used as supply to meet that region's demand.
    # indexed the same as capacity_from_region
    capacity_kept = [m.addMVar((n_regions))
                     for _ in range(n_periods)]
    for i in range(n_periods):
        for speed_idx, speed in enumerate(charging_capacity.keys()):
            m.addConstr(capacities[i][speed_idx] == (chargers_added_at_region[speed][i] + existing_chargers[speed_idx])
                        * charging_capacity[speed])

        for j in range(n_regions):
            # CONSTRAINT: Maximum number of charging stations in a single region
            if (max_chargers_per_region > 0):
                m.addConstr(
                    chargers_added_at_region[speed][i][j] <= max_chargers_per_region)

            for k in range(n_regions):
                m.addConstr(capacity_to_neighbors[i][j][k] >= 0)
                # the below constraint is to eliminate pointless, unrealistic two-way exchanges of energy
                # in that it forces one direction to be 0.
                m.addConstr(
                    capacity_to_neighbors[i][j][k]*capacity_to_neighbors[i][k][j] == 0)

            m.addConstr(capacity_kept[i][j] >= 0)
            # CONSTRAINT: MAX TOTAL ADDED ENERGY IN ONE SQUARE
            if max_supply_per_region > 0:
                m.addConstr(supply[i][j] <= max_supply_per_region)
            m.addConstr(capacity_from_region[i] == sum(capacities[i]))

            # this one is doing most of the heavy lifting, simply partitioning the energy produced
            # into two expressions, the amount kept and the sum of the amounts donated to its neighbors
            m.addConstr(sum([capacity_to_neighbors[i][j][k-1] for k in get_neighbors(
                j, demand_data)]) + capacity_kept[i][j] == capacity_from_region[i][j])

            # the total energy supply of a region is the amount of energy produced by the region kept
            # + the total sum of energy donated by its neighbors
            m.addConstr(supply[i][j] == capacity_kept[i][j] + sum(
                [capacity_to_neighbors[i][k-1][j] for k in get_neighbors(j, demand_data)]))

        # keeping count of the total number of chargers in each year, for the possibility of limiting this number in the solution
        # and for metrics
        for j in range(len(charging_capacity.keys())):
            m.addConstr(number_of_chargers_year[i][j] == np.array([np.array(
                [chargers_added_at_region[speed][i][j] for j in range(n_regions)]).sum() for speed in charging_capacity.keys()][j]))

    # we use this as the basis for our objective value
    diff = m.addMVar(supply.shape, ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)
    # this norms variable lets us analyse the distance between supply and demand,
    # rather than just the difference.
    # index norms : norms[i] is the norm of the difference between supply and demand on year i
    norms = m.addMVar(n_periods)
    for i in range(n_periods):
        m.addConstr(diff[i] == supply[i]-demand[i])
        m.addConstr(norms[i] == gp.norm(diff[i], 1))
        for j in interest_points["grid number"]:
            m.addConstr(supply[i][j] >= POI_importance*demand[i][j])
    total_chargers = m.addVar()
    m.addConstr(total_chargers == sum([sum(number_of_chargers_year[i])
                                       for i in range(n_periods)]))

    m.setObjective(gp.quicksum(norms), gp.GRB.MINIMIZE)
    # +sum(number_of_chargers_year[n_periods-1])*10000

    m.setParam("MIPGap", MIP_gap)
    m.setParam("TimeLimit", time_limit)
    m.optimize()

    with open("results.txt", "w") as f:
        sys.stdout = f
        # print(supply.shape)
        for i in range(n_periods):
            print('Year: {}'.format(i))
            if i >= 1:
                for speed_idx, speed in enumerate(charging_capacity.keys()):
                    print(
                        "Add", int(number_of_chargers_year[i][speed_idx].X - number_of_chargers_year[i-1][speed_idx].X), speed)
            else:
                for speed_idx, speed in enumerate(charging_capacity.keys()):
                    print(
                        "Add", int(number_of_chargers_year[i][speed_idx].X), speed)
            print("Year Total: {}".format(
                int(sum(number_of_chargers_year[i].X))))
            for j in range(n_regions):
                # if not (demand[i][j] == 0 and supply[i][j].X == 0):
                print(
                    f"Region {j}: \n\tDemand: {demand[i][j]} \n\tSupply: {supply[i][j].X}\n\tSupplying {capacity_from_region[i][j].X}kWh/year \n\tSupplying to: {[(k-1, capacity_to_neighbors[i][j][k-1].X) for k in get_neighbors(j, demand_data)]} \n\tSupplied by: {[(k-1, capacity_to_neighbors[i][k-1][j].X) for k in get_neighbors(j, demand_data)]} \n\tKeeping {capacity_kept[i][j].X}")
                # elif (capacity_from_region[i][j].X > 0) or (any(capacity_to_neighbors[i][j].X > 0)):
                # sys.stdout = sys.__stdout__
                # print(
                #     f"non-zero stuff at year {i} region {j}, {capacity_from_region[i][j].X} or {capacity_to_neighbors[i][j].X}")
                # sys.stdout = f

        print(m.getObjective().getValue())

        sys.stdout = sys.__stdout__

    for i in range(n_periods):
        demand_data["Supply_{}".format(i)] = supply[i].X
        # demand_data["Diff_{}".format(i)] = [abs(min(0, x))
        #                                     for x in (diff[i].X)]
        demand_data["Diff_{}".format(i)] = [abs(x) for x in (diff[i].X)]
    return demand_data


# get_new_dataframe(d, p)
