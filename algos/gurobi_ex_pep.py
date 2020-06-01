# -*- coding: utf-8 -*-
"""
modifed on Fri May 29 13:25:07 2020

imported from ipynb on Mon May 25 12:21:32 2020

Super route finder:
    huristic optimization for finding super routes

@author: Sal
"""

from gurobipy import * 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import copy



Length = 20000 # a constant for the maximum of length of a route
n_zones = 71 # number of zones in the area
zones=[]
for i in range(1,71): # ceating zones or nodes representing super stops
    zones.append(str(i))



#importing the links of the graph, defining the relationship and connections between each zone
Contingencies = pd.read_csv("C:\\Users\\Novin\\Desktop\\File_Transfer\\Contingencies.csv")
# creating the arcs as a tuple dict for guropbi
distance = tupledict()# tupledicts are gurobi objects that allow systematic indexing of variables

for i in range(0,len(Contingencies)):
    distance[(str(Contingencies["src_SS_GROUP"][i]),str(Contingencies["nbr_SS_GROUP"][i]))] = Contingencies ["distance"][i]
arc = distance.keys()


#importing the direct distance between each node, this information will later be used to calculate route's circuity
OD_Distances = pd.read_csv("C:\\Users\\Novin\\Desktop\\File_Transfer\\K_Means70_OD_Distance.csv")
OD_Direct_Dist = tupledict()
for i in range(0,len(OD_Distances)):
    OD_Direct_Dist[(str(OD_Distances["Origin"][i]),str(OD_Distances["Destination"][i]))] = OD_Distances ["DISTANCE"][i]
    

# importing the demand matrix
OD_Matrix = pd.read_csv("C:\\Users\\Novin\\Desktop\\File_Transfer\\Peak_Period_OD_K_Means_70.csv")
OD_Matrix.rename(columns={'Unnamed: 0':'Origins'}, inplace=True)
OD_Matrix = OD_Matrix.set_index("Origins")
OD_Matrix.index = OD_Matrix.index.map(str)


#creating the demand for graph
demand = tupledict()
for i in OD_Matrix:
    for j in OD_Matrix.columns: 
        demand [(i,j)] = OD_Matrix[i][j]
ODs = demand.keys() 

#since route are symmetric, we take away half of the ods that are repetetive
half_ODs = copy.deepcopy(ODs)
for i in ODs:
    if int(i[0])>int(i[1]):
        half_ODs.remove(i)

# optimization part
###############3   
def path_finder(start,end,size,circuity,length,distances):
    
    '''
    This function finds the path that directly serves maximum demand, given a starting point and an ending point 
    and a route circuity. This function is the heuristic that will make the feasible region of the super route optmization smaller
    by fixing the two ends of the route and then finding the super route; rather than having the optimization do both at the same time.
    another feture is the variable size that governs the number of zones that will be in the found route
    
    Inputs:
        start:  the node where the route will start
        end : the node where the route will end
        size:  the number of zones included in the route, ie. belong.sum()
        circuity: variable constraining the maximum circuity of a the route to be found
        lenght: variable constraining the maximum lenghth of the route to be found
        distances: a dictionary containing the length of each link on the network
        
    Outputs : 
        a list of lists containing [the objective function value, meaning the served demand, the zones served by the path, and the link that construct the path]
    
    out of scope variables: the dictionary OD_Direct_Distance from the main script     
    '''
    s=start
    e=end
    c=circuity
    l=length
    n=size
    d=distances
    m = Model ("super_route_finder")
    OD_on_route = m. addVars ( half_ODs, name =" OD_on_route ",vtype=GRB.BINARY)
    starts = m. addVars ( zones , name ="start",vtype=GRB.BINARY)
    ends = m. addVars ( zones , name ="end",vtype=GRB.BINARY)
    path = m. addVars ( arc, name ="path",vtype=GRB.BINARY)
    belong = m. addVars ( zones, name ="belong",vtype=GRB.BINARY)
    yous = m. addVars ( zones, name ="yous",vtype=GRB.INTEGER)
    m. addConstr (path.prod(d)<=l, name = "length")
    m. addConstr (( starts.sum() == 1), "one_start ")
    m. addConstr (( ends.sum() == 1), "one_end ")
    #m. addConstrs ((( starts[i] + ends[i]) <= 1 for i in zones),"nifu_nifa")
    m. addConstrs (( path. sum (j,'*') - path. sum ('*',j) == starts[j]-ends[j] for j in zones) , "conservation ")
    m. addConstrs (( belong[i]== (path.sum(i,'*')+ ends[i])for i in zones),"does_belong?")
    m. addConstr (( belong.sum() == n), "route size ")
    m. addConstr (( starts[s] == 1), "specific_start ")
    m. addConstr (( ends[e] == 1), "specific_end ")
    m. addConstrs ((OD_on_route[(i,j)] <= belong[str(i)] for i,j in OD_on_route ))
    m. addConstrs ((OD_on_route[(i,j)] <= belong[str(j)] for i,j in OD_on_route ))
    m. addConstrs ((OD_on_route[(i,j)] >= belong[str(i)]+belong[str(j)]-1 for i,j in OD_on_route ))
    m. addConstrs (( yous[i]>=0 for i in zones), "you_declaration_1 ")
    m. addConstrs (( yous[i] +1<=n for i in zones), "you_declaration_2 ")
    m. addConstrs (( yous[i]-yous[j] + n* path[(i,j)]+1 <=n for i,j in arc if j !=i), "subtour elim")
    #m. addConstrs ((path.prod(distance)<=(0.5*C*OD_Direct_Dist[(r,s)] * (starts[r]+ends[s]) +( 2 - starts[r] -ends[s])* 10000000)  for r,s in OD_on_route if r !=s), name = "Circuity")
    m. addConstr ((path.prod(d)<=(c*OD_Direct_Dist[(s,e)])),"circuity ")
    m.setParam(GRB.Param.OutputFlag,0) # controls the logging and printing
    m.setParam(GRB.Param.MIPGap,0.05) # controls the threshold for ending the optimization
    temp = LinExpr() # gurobi object allowing for linear expressions of variables
    for i,j in half_ODs:
        temp.add(OD_on_route[i,j],(demand[i,j]+demand[j,i]))
    m.setObjective(temp,GRB.MAXIMIZE)
    m.update()
    m.optimize ()
    my_path = []
    my_zones = []
    if m.status == 2 :
        print("("+s+","+e+")"+"_"+str(n)+"=",m.objval)
        solution = m. getAttr ("x", path)
        for i,j in solution:
            if solution[i,j]==1:
                my_path.append((i,j))
        solution = m. getAttr ("x", belong)
        for i in solution:
            if solution[i]==1:
                my_zones.append(i)        
        return [m.objval,my_zones,my_path]
    else:
        return 0
    
#################
        
#main function
import timeit
start_time = timeit.default_timer()
routes = {}
for i,j in half_ODs:
    if i!=j:
        if OD_Direct_Dist[(i,j)]>9000:# only chooses starts and ends that are farther than 9000m from each other
            for size in range(20,30):
                routes[(i,j,size)]= path_finder(i,j,size,1.5,20000,distance)
elapsed = timeit.default_timer() - start_time
print(elapsed)  

#####

# post processing
def Joint_coverage(r1,r2,demand):
    '''
    calculates the total demand that can be served directly by two routes, r1,r2 
    input: 
        r1 = route1
        r2 = route2
        demand: od matrix
    output: 
        the value of served demand
    '''
    total_coverage = r1[0]+r2[0]
    double_coverage=0
    all_zones=list(set(r1[1])& set(r2[1]))
    for i in all_zones:
        for j in all_zones:
            double_coverage= double_coverage + demand[i,j]
            
    return total_coverage-double_coverage

###
start_time = timeit.default_timer()
count = 0
multi_routes={}
for r1 in routes:
    if routes[r1]!=0:
        for r2 in routes:
            if routes[r2]!=0:
                multi_routes[r1,r2]= Joint_coverage(routes[r1],routes[r2],demand)
                count +=1 
                print(count)
elapsed = timeit.default_timer() - start_time
print(elapsed)    

max (multi_routes,key = multi_routes.get)
max (multi_routes.values())


