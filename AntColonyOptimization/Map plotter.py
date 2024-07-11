import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
#%%

tree = ET.parse('PycharmProjects/2.kml')
root = tree.getroot()


points = np.array([[2.12,0.7],[1.68,0.97],[1.28,1.5],[1.16,2.3],[1.12,4],
                  [1.2,5.4],[1.32,7.2],[1.64,8.4],[1.92,8.6],[2.2,8],
                  [2.52,7.6],[2.72,6.4],[2.8,4.8],[2.72,3.3],[2.56,2.3],[2.4,1.2]])
#points.shape
plt.scatter(points[:,0],points[:,1]);

#x = []
#y = []
arr = np.empty((0,2))

for i,placemark in enumerate(root.iter('{http://www.opengis.net/kml/2.2}Placemark')):
    # Extract relevant data from placemark
    name = placemark.find('{http://www.opengis.net/kml/2.2}name').text
    coordinates = placemark.find('{http://www.opengis.net/kml/2.2}Point/{http://www.opengis.net/kml/2.2}coordinates').text
    
    # Add x,y coordinates to arrays 
    x = float(coordinates.split(',')[0].split(" ")[-1])
    y = float(coordinates.split(',')[1])
    arr = np.append(arr,[[x,y]],axis=0)
    
    #x.append(float(coordinates.split(',')[0].split(" ")[-1]))
    #y.append(float(coordinates.split(',')[1]))

#%%
# Plotting 10 first datapoints 
plt.scatter(arr[:10,0],arr[:10,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Coordinate Plot')

# Adding annotations for each point
for i, txt in enumerate(arr[:10,0]):
    plt.annotate(i+1, (arr[i][0], arr[i][1]), textcoords='offset points', xytext=(0, 10), ha='center')

plt.show()

#%%
def eucl_dist(x,y):
    
    summation = 0
    
    for dim in np.arange(len(x)):
        summation += (x[dim] - y[dim])**2
    
    return np.sqrt(summation)

def path_length(array):
    
    length = 0
    
    for i in  np.arange(array.shape[0]-1):
        length += eucl_dist(array[i],array[i+1])
        
    length += eucl_dist(array[-1],array[0])   
    
    return length

def path_lenght_impr(array):
    differences = np.diff(array, axis=0)  # Calculate the differences between consecutive points
    distances = np.linalg.norm(differences, axis=1)  # Calculate the Euclidean distances
    length = np.sum(distances)  # Sum up the distances
    length += np.linalg.norm(array[-1] - array[0])  # Add the distance between the last and first points
    return length

def distance_matrix(points):
    
    points_size = points.shape[0]
    dist_mat = np.zeros((points_size,points_size))
    
    for i in range(points_size):
        for j in range(points_size):
            dist_mat[i,j] = eucl_dist(points[i],points[j])
            
    return dist_mat

def distance_matrix_impr(points):
    points_size = points.shape[0]
    dist_mat = np.zeros((points_size, points_size))
    
    for i in range(points_size):
        for j in range(i+1, points_size):  # Iterate only over the upper triangular part
            dist = eucl_dist(points[i], points[j])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist  # Copy the value to the lower triangular part
    
    return dist_mat

#%%
dist_map = distance_matrix_impr(arr)
sns.heatmap(dist_map)

#%%
def inverse_distance_matrix(points):
    
    points_size = points.shape[0]
    inv_dist_mat = np.zeros((points_size,points_size))
    
    # first, construct the distance matrix
    dist_mat = distance_matrix(points)
    
    for i in range(points_size):
        for j in range(points_size):
            if i==j:
                # for the diagonal the inverse distance is not defined, thus set 0.0
                inv_dist_mat[i,j] = 0.0
            else:
                inv_dist_mat[i,j] = 1.0/dist_mat[i,j]
            
    return inv_dist_mat

def inverse_distance_matrix_impr(points):
    points_size = points.shape[0]
    
    # Calculate distance matrix using vectorized operations
    dist_mat = np.linalg.norm(points[:, np.newaxis] - points, axis=-1)
    
    # Create an identity matrix and calculate inverse distances using broadcasting
    inv_dist_mat = np.where(np.eye(points_size, dtype=bool), 0.0, 1.0 / dist_mat)
    
    return inv_dist_mat

#%%

inv_dist_mat = inverse_distance_matrix_impr(arr)
sns.heatmap(inv_dist_mat)

#%%

class Ant:
    def __init__(self,n_locations):
        
        self.n_locations = n_locations
        self.position = np.random.choice(n_locations)
        
        self.places_visited = [self.position]
        self.places_left = list(np.arange(n_locations))
        self.places_left.remove(self.position)
        
        self.phero_graph = np.full((n_locations,n_locations),0.0)
        self.travel_probas = np.zeros(n_locations-1)
        
        self.tour_cost = 0.0
        
        
    def ant_trip(self,g_phero_graph,dist_mat,inv_dist_mat,alpha=1,beta=1,Q=1):

        for i in np.arange(len(self.places_left)):
            # is index from 0 to places_left-1
            
            # determine the probabilities for next move
            #------------------------------------------

            # compute numerator and denominator for proba-calculating fractions
            allowed_weights = []
            for loc in self.places_left:
                #vector with numerators for each allowable move
                allowed_weights.append((g_phero_graph[self.position,loc]**alpha)*(inv_dist_mat[self.position,loc]**beta))
                
            # this is the denominator of the proba-calculating fraction
            allowed_weights_sum = np.sum(allowed_weights)
            
            # probabilities for next move
            
            travel_probas = allowed_weights/allowed_weights_sum
            
            
            #stochastically pick next destination
            #------------------------------------
            next_destination = np.random.choice(self.places_left,p=travel_probas)

            # update info
            #------------
            
            # add distance into total distance travelled
            self.tour_cost += dist_mat[self.position,next_destination]
            
            # then change position and update travel-log variables
            self.position = next_destination
            self.places_visited.append(next_destination)
            self.places_left.remove(next_destination)
    

        # after it has finished, update self.phero_graph
        for i,j in zip(self.places_visited[:-1],self.places_visited[1:]):
            self.phero_graph[i,j] = Q/self.tour_cost
            
            
    def ant_flush(self):
        
        self.position = np.random.choice(self.n_locations)
        
        self.places_visited = [self.position]
        self.places_left = list(np.arange(self.n_locations))
        self.places_left.remove(self.position)
        
        self.phero_graph = np.full((self.n_locations,self.n_locations),0.0)
        self.travel_probas = np.zeros(self.n_locations-1)
        
        self.tour_cost = 0.0
        

#%%
def update_pheromones(g_phero_graph,ants,evapo_coef=0.05):

    dim = g_phero_graph.shape[0]
    
    for i in range(dim):
        for j in range(dim):
            g_phero_graph[i,j] = (1 - evapo_coef)*g_phero_graph[i,j] + np.sum([ant.phero_graph[i,j] for ant in ants])
            g_phero_graph[i,j] = max(g_phero_graph[i,j],1e-08) # avoid division by zero
            
    return g_phero_graph            


#%%
def aco(points,alpha,beta,evapo_coef,colony_size,num_iter):
    
    # compute (once) the distance matrices
    dist_mat = distance_matrix(points)
    inv_dist_mat = inverse_distance_matrix(points)
    
    n_locations = points.shape[0] # total number of points
    ants = [Ant(n_locations) for i in range(colony_size)] # ant colony
    
    # determine initial pheromone value
    phero_init = (inv_dist_mat.mean())**(beta/alpha)
    g_phero_graph = np.full((n_locations,n_locations),phero_init) # pheromone matrix (arbitrary initialization)
    
    # determine scaling coefficient "Q"
    [ant.ant_trip(g_phero_graph,dist_mat,inv_dist_mat,1) for ant in ants]
    best_ant = np.argmin([ant.tour_cost for ant in ants]) #ant that scored best in this iteration
    Q = (ants[best_ant].tour_cost)*phero_init/(0.1*colony_size)
    print(Q)
        
    best_path_length = ants[best_ant].tour_cost
    best_path = ants[best_ant].places_visited.copy()
    
    monitor_costs = []    

    for i in tqdm(np.arange(num_iter)):
    
        [ant.ant_trip(g_phero_graph = g_phero_graph,dist_mat = dist_mat,inv_dist_mat = inv_dist_mat,Q = Q) for ant in ants]
        g_phero_graph = update_pheromones(g_phero_graph,ants,evapo_coef).copy()
    
        iteration_winner = np.argmin([ant.tour_cost for ant in ants]) #ant that scored best in this iteration
        best_path_iteration = ants[iteration_winner].places_visited
        
        # update global best if better
        if best_path_length > ants[iteration_winner].tour_cost:
            best_ant = iteration_winner
            best_path = best_path_iteration.copy()
            best_path_length = ants[iteration_winner].tour_cost
            
       # monitor_costs.append(ants[iteration_winner].tour_cost)
        monitor_costs.append(best_path_length)
        
        [ant.ant_flush() for ant in ants]
        
    return best_path,monitor_costs


#%%
best_path, monitor_costs = aco(points = arr, 
                               alpha = 1, 
                               beta = 1, 
                               evapo_coef = 0.05, 
                               colony_size = 40, 
                               num_iter = 100)
print (monitor_costs[-1])
print('')

plt.figure(figsize=(20,5));
plt.subplot(1,2,1);
plt.plot(arr[best_path,0],
         arr[best_path,1]);


plt.subplot(1,2,2);
plt.plot(np.arange(len(monitor_costs)),monitor_costs);
