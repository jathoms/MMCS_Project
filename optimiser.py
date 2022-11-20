from distutils.ccompiler import new_compiler
import geopandas as gpd
import pandas as pd
import numpy as np
# import geoplot as gplots
import matplotlib.pyplot as plt
import gurobipy as gp

d = pd.read_excel("./data/Demand_data.xlsx")
d['geometry'] = gpd.GeoSeries.from_wkt(d['geometry'])
gd = gpd.GeoDataFrame(d, geometry='geometry', crs='EPSG:4326')

n_periods = 4
n_regions = len(d["Ref"])


def get_period_names(upto_n: int):
    if upto_n > 4:
        print("N too high, must be integer <= 4")
        return None
    return ["Demand_{}".format(p) for p in range(upto_n)]


def get_neighbors(region, data):
    return [int(x) for x in data['NEIGHBORS'][region].strip('][').split(', ')]


def get_new_dataframe(demand_data: pd.DataFrame):

    m = gp.Model()
    demand = np.array([demand_data[period]
                       for period in get_period_names(n_periods)])
    # maximum rates for each type of charger in kWh/year
    charging_capacity = {'slow': 3500, 'fast': 5200, 'rapid': 50500}

    supply = m.addMVar(demand.shape)

    chargers_at_region = {
        p: m.addMVar(demand.shape, vtype=gp.GRB.INTEGER) for p in charging_capacity.keys()}
    number_of_chargers_year = m.addMVar(
        (n_periods, len(charging_capacity.keys())))
    # indexing chargers at region:
    # chargers_at_region['fast'][0][2] to get the number of 'fast' chargers during time period 0 in region 3

    ne = m.addMVar(n_regions, vtype=gp.GRB.BINARY)
    for i in range(n_periods):
        capacities = [slow_capacity, fast_capacity, rapid_capacity] = [(chargers_at_region[speed][i]
                                                                        * charging_capacity[speed]) for speed in charging_capacity.keys()]
        capacity_from_region = np.array(capacities).sum()
        # neighbor_binaries = np.zeros((capacity_from_region.shape))
        # for j in range(n_regions):
        #     for k in get_neighbors(j, demand_data):
        #         neighbor_binaries[k] = 1
        #     print(neighbor_binaries.shape)
        #     print(capacity_from_region.shape)
        #     #   (ne @ fast_capacity) + (ne @ rapid_capacity))
        m.addConstr(
            supply[i] == capacity_from_region)  # + (neighbor_binaries * capacity_from_region))

        # capacity_from_neighbors = np.array(
        #     [sum([capacity_from_region[k] for k in get_neighbors(j, demand_data)]) for j in range(n_regions)])

        for j in range(len(charging_capacity.keys())):
            m.addConstr(number_of_chargers_year[i][j] == np.array([np.array(
                [chargers_at_region[speed][i][j] for j in range(n_regions)]).sum() for speed in charging_capacity.keys()][j]))

    diff = m.addMVar(supply.shape, ub=100, lb=-gp.GRB.INFINITY)

    m.addConstrs((diff[i] == supply[i]-demand[i] for i in range(n_periods)))
    total_chargers = m.addVar()
    m.addConstr(total_chargers == sum([sum(number_of_chargers_year[i])
                                       for i in range(n_periods)]))
    norms = m.addMVar(n_periods)
    m.addConstrs((norms[i] == gp.norm(diff[i], 1) for i in range(n_periods)))
    m.setObjective(gp.quicksum(norms) + (total_chargers)*1000, gp.GRB.MINIMIZE)
    m.optimize()

    # print(supply.shape)
    # for i in range(n_periods):
    #     print('Year: {}'.format(i))
    #     print(number_of_chargers_year[i].X)
    #     print("Year Total: {}".format(sum(number_of_chargers_year[i].X)))
    # print("Total total: {}".format(
    #     sum([sum(number_of_chargers_year[i].X) for i in range(n_periods)])))
    for i in range(n_periods):
        demand_data["Supply_{}".format(i)] = supply[i].X
        demand_data["Diff_{}".format(i)] = [abs(x) for x in (diff[i].X)]

    return demand_data


get_new_dataframe(d)
