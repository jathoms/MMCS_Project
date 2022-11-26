import geopandas as gpd
import pandas as pd
import numpy as np
import gurobipy as gp
from shapely import wkt
import sys

demand_data = pd.read_excel("./data/Demand_data.xlsx")
demand_data['geometry'] = demand_data['geometry'].apply(wkt.loads)
interest_points = pd.read_excel("./data/Interest_points.xlsx")
interest_points['geometry'] = interest_points['geometry'].apply(wkt.loads)
potential_df = pd.read_excel("./data/Potential_charging_points.xlsx")
potential_df['geometry'] = potential_df['geometry'].apply(wkt.loads)
charging_points_df = pd.read_excel("./data/Charging_points.xlsx")
charging_points_df['geometry'] = charging_points_df['geometry'].apply(
    wkt.loads)
sys.stdout = sys.__stdout__


def get_period_names(upto_n: int):
    if upto_n > 4:
        print("N too high, must be integer <= 4")
        return None
    return ["Demand_{}".format(p) for p in range(upto_n)]


n_periods = 4
n_regions = len(demand_data["Ref"])

# maximum rates for each type of charger in kWh/year. assuming constant maximum usage for now.
charging_capacity = {'Slow': 2750, 'Fast': 4600, 'Rapid': 40250}

existing_chargers = np.array([demand_data["Number of {} Charging Points".format(
    speed)] for speed in charging_capacity.keys()])
region_distance_from_cc = np.array(demand_data["Distance from Centre"])

potential_points_per_region = np.array(
    demand_data["Number of Potential Locations"])

# demand[i][j] is the demand (in kWh/year) on year i of region j
demand = np.array([demand_data[period]
                   for period in get_period_names(n_periods)])

# certain adjustable parameters
max_chargers_per_region = 0
max_added_chargers = 100
max_supply_per_region = 100000
min_avg_distance_from_cc = 500
POI_importance = 0
rapid_supply_limit = 0.5

time_limit = 60
MIP_gap = 0.05

m = gp.Model()


def get_neighbors(region, data):
    # we need this to access the neighbors of any given region, \
    # converts from string rep. of a list "[0,1,2]" to an actual list [0,1,2]
    return [int(x) for x in data['NEIGHBORS'][region].strip('][').split(', ')]


# supply[i][j] is same as demand, but represents supply
supply = m.addMVar(demand.shape)


# main decision variable
# indexing chargers at region:
# chargers_at_region['fast'][0][2] to get the number of 'fast' chargers during time period 0 in region 3
chargers_added_at_region = [m.addMVar((len(charging_capacity.keys()), n_regions), vtype=gp.GRB.INTEGER)
                            for _ in range(n_periods)]

# simple sum over every type of charger over every region for each time period
number_of_chargers_added_year = m.addMVar(
    (n_periods, len(charging_capacity.keys())), vtype=gp.GRB.INTEGER)

# a ['slow', 'fast', 'rapid'] shape for each region, indicating how much energy
# is supplied by a specific source to each region
chargers_at_region = [m.addMVar((len(charging_capacity.keys()), n_regions), vtype=gp.GRB.INTEGER)
                      for _ in range(n_periods)]

# sum over the previous variable's speeds, not including/including the pre-existing chargers respectively
total_chargers_at_region = m.addMVar(supply.shape, vtype=gp.GRB.INTEGER)
total_chargers_added_at_region = m.addMVar(supply.shape, vtype=gp.GRB.INTEGER)

capacities = [m.addMVar((len(charging_capacity.keys()), n_regions))
              for _ in range(n_periods)]

# indicating the amount of energy supplied BY a region on each year (from chargers inside that region)
# index capacity_from_region[i][j] : capacity from region j on year i
capacity_from_region = [m.addMVar((n_regions))
                        for _ in range(n_periods)]

# indicating how much energy can potentially be used by a region's neighbors
# index capacity_from_region[i][j][k] : energy from region j used by region k on year i
capacity_to_neighbors = [m.addMVar((n_regions, n_regions))
                         for _ in range(n_periods)]

for speed in range(len(charging_capacity.keys())):
    for i in range(n_periods-1):
        for j in range(n_regions):
            # enforcing continuity, in that the number of chargers can only increase year-on-year.
            # without this constraint, our configuration can look completely different between years
            m.addConstr(
                chargers_added_at_region[i+1][speed][j] >= chargers_added_at_region[i][speed][j])

for i in range(n_periods):
    for speed_idx, speed in enumerate(charging_capacity.keys()):
        m.addConstr(
            chargers_at_region[i][speed_idx] == chargers_added_at_region[i][speed_idx] + existing_chargers[speed_idx])

        m.addConstr(capacities[i][speed_idx] == (chargers_at_region[i][speed_idx])
                    * charging_capacity[speed])

    m.addConstr(total_chargers_at_region[i] == sum(
        chargers_at_region[i]))

    m.addConstr(total_chargers_added_at_region[i] == sum(
        chargers_added_at_region[i]))

    m.addConstr(min_avg_distance_from_cc*sum(total_chargers_added_at_region[i]) <= gp.quicksum(
        [total_chargers_added_at_region[i][j] * region_distance_from_cc[j] for j in range(n_regions)]))

    # m.addConstr(avg_distance_from_cc[i] >= min_avg_distance_from_cc)
    for j in range(n_regions):
        # CONSTRAINT: Maximum number of charging stations in a single region
        if (max_chargers_per_region > 0):
            m.addConstr(
                total_chargers_added_at_region[i][j] <= max_chargers_per_region)
        m.addConstr(
            total_chargers_added_at_region[i][j] <= potential_points_per_region[j])

        for k in range(n_regions):
            m.addConstr(capacity_to_neighbors[i][j][k] >= 0)
            # the below constraint is to eliminate pointless, unrealistic two-way exchanges of energy
            # in that it forces one direction to be 0.
            m.addConstr(
                capacity_to_neighbors[i][j][k]*capacity_to_neighbors[i][k][j] == 0)

        # m.addConstr(capacity_kept[i][j] >= 0)
        m.addConstr(capacity_to_neighbors[i][j][j] >= 0)
        # CONSTRAINT: MAX TOTAL ADDED ENERGY IN ONE SQUARE
        if max_supply_per_region > 0:
            m.addConstr(supply[i][j] <= max_supply_per_region)
        m.addConstr(capacity_from_region[i] == sum(capacities[i]))

        # this one is doing most of the heavy lifting, simply partitioning the energy produced
        # into two expressions, the amount kept and the sum of the amounts donated to its neighbors
        m.addConstr(sum([capacity_to_neighbors[i][j][k-1] for k in get_neighbors(
            j, demand_data)]) + capacity_to_neighbors[i][j][j] == capacity_from_region[i][j])

        # the total energy supply of a region is the amount of energy produced by the region kept
        # + the total sum of energy donated by its neighbors
        m.addConstr(supply[i][j] == capacity_to_neighbors[i][j][j] + sum(
            [capacity_to_neighbors[i][k-1][j] for k in get_neighbors(j, demand_data)]))

    # keeping count of the total number of chargers in each year, for the possibility of limiting this number in the solution
    # and for metrics
    for j in range(len(charging_capacity.keys())):
        m.addConstr(number_of_chargers_added_year[i][j] == np.array([np.array(
            [chargers_added_at_region[i][speed_idx][j] for j in range(n_regions)]).sum() for speed_idx in range(len(charging_capacity.keys()))][j]))


# max chargers at the end.
if max_added_chargers >= 0:
    m.addConstr(sum(number_of_chargers_added_year[3]) <=
                max_added_chargers)

# we use this as the basis for our objective value
diff = m.addMVar(supply.shape, ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)
# this norms variable lets us analyse the distance between supply and demand,
# rather than just the difference.
# index norms : norms[i] is the norm of the difference between supply and demand on year i
norms = m.addMVar(n_periods)
for i in range(n_periods):
    m.addConstr(sum(capacities[i][2]) <=
                rapid_supply_limit*sum(supply[i]))
    m.addConstr(diff[i] == supply[i]-demand[i])
    m.addConstr(norms[i] == gp.norm(diff[i], 1))
    for j in interest_points["grid number"]:
        m.addConstr(sum([supply[i][j] for j in interest_points["grid number"]]) >= POI_importance*sum([demand[i][j]
                    for j in interest_points["grid number"]]))
m.setObjective(gp.quicksum(norms), gp.GRB.MINIMIZE)

# defaults, supposed to be changed by run_model()
global_max_supply = 100000.0
global_max_demand = 100000.0
global_max = 100000.0


def run_model():
    m.setParam("MIPGap", MIP_gap)
    m.setParam("TimeLimit", time_limit)
    m.optimize()

    global global_max_supply
    global_max_supply = max(
        [max([supply[i][j].X for j in range(n_regions)]) for i in range(n_periods)])
    global global_max_demand
    global_max_demand = max(
        [max([demand[i][j] for j in range(n_regions)]) for i in range(n_periods)])
    global global_max
    global_max = max(global_max_demand, global_max_supply)
    print('done')


def print_results():
    avg_distance_from_cc = [sum(
        [total_chargers_added_at_region[i][j].X * region_distance_from_cc[j] for j in range(n_regions)])/sum(total_chargers_added_at_region[i].X) for i in range(n_periods)]
    with open("results.txt", "w") as f:
        sys.stdout = f
        for i in range(n_periods):
            print('Year: {}'.format(i))
            print(
                f'Avg distance of added chargers from city centre: {avg_distance_from_cc[i]}')
            if i >= 1:
                for speed_idx, speed in enumerate(charging_capacity.keys()):
                    print(
                        "Add", int(number_of_chargers_added_year[i][speed_idx].X - number_of_chargers_added_year[i-1][speed_idx].X), speed)
            else:
                for speed_idx, speed in enumerate(charging_capacity.keys()):
                    print(
                        "Add", int(number_of_chargers_added_year[i][speed_idx].X), speed)
            print("Year Total: {}".format(
                int(sum(number_of_chargers_added_year[i].X))))

            for j in range(n_regions):
                # for speed_idx, speed in enumerate(charging_capacity.keys()):
                #     print(f'{speed}: {capacities[i][speed_idx].X}')
                # if not (demand[i][j] == 0 and supply[i][j].X == 0):
                print(
                    f"Region {j}: \n\tDemand: {demand[i][j]} \n\tSupply: {supply[i][j].X}\n\tSupplying {capacity_from_region[i][j].X}kWh/year \n\tSupplying to: {[(k-1, capacity_to_neighbors[i][j][k-1].X) for k in get_neighbors(j, demand_data)]} \n\tSupplied by: {[(k-1, capacity_to_neighbors[i][k-1][j].X) for k in get_neighbors(j, demand_data)]} \n\tKeeping {capacity_to_neighbors[i][j][j].X}")
                # elif (capacity_from_region[i][j].X > 0) or (any(capacity_to_neighbors[i][j].X > 0)):
                # sys.stdout = sys.__stdout__
                # print(
                #     f"non-zero stuff at year {i} region {j}, {capacity_from_region[i][j].X} or {capacity_to_neighbors[i][j].X}")
                # sys.stdout = f

        print(m.getObjective().getValue())

        sys.stdout = sys.__stdout__


def get_supply_diff_dataframe():

    for i in range(n_periods):
        demand_data["Supply_{}".format(i)] = supply[i].X
        demand_data["Diff_{}".format(i)] = [abs(min(0, x))
                                            for x in (diff[i].X)]
        # demand_data["Diff_{}".format(i)] = [abs(x)
        #                                     for x in (diff[i].X)]
    return demand_data


# get_new_dataframe(d, p)
