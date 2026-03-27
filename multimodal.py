import numpy as np


def select_clusters(size_pop, n, m, count_of_clusters, uplimit,lowlimit, fofma):
    # n - длина вектора переменных
    # m - число ближайших соседей
    def add_column(data,x,z):
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                if (j == 0 or j==1):
                    data[i][j]=x[i][j]
                else:
                    data[i][j]=z[i]
        return data
    def give_min_m(m,distanses):
        distanses = distanses[distanses[:,2].argsort()]
        return distanses[0:m]
    def remove_cluster(x,cluster):
        to_remove = []
        for i in range(len(cluster)):
            for j in range(len(x)):
                if (np.all(cluster[i]==x[j])):
                    to_remove.append(j)
        x=np.delete(x,to_remove,0)
        return x
    list_of_clusters = []
    np.random.seed(10)
    x=np.random.uniform(uplimit,lowlimit, (size_pop,n))
    if count_of_clusters == 1: return x
    #print(f'{x=}')
    for i in range(count_of_clusters):
        z = []
        for i in range(size_pop):
            z.append(fofma(x[i]))
        data = np.ones((len(z),n+1))
        data = add_column(data,x,z)
        # for i in range(size_pop):
        #     for j in range(n+1):
        #         if (j == 0 or j==1):
        #             data[i][j]=x[i][j]
        #         else:
        #             data[i][j]=z[i]

        data = data[data[:,2].argsort()]
        data_set = data.copy()
        #print(data_set)
        superior = data_set[0,0:n]
        #print(f'{superior=}')

        distanses = np.delete(data_set, 0,0)
        #print(f'{distanses=}')
        for i in range(len(distanses)):
            distanses[i][n] = (np.linalg.norm(superior-distanses[i][0:n]))
        #print(f'{distanses=}')
        cluster = give_min_m(m,distanses)
        #print(f'{cluster=}')
        cluster = np.delete(cluster, 2,1)       # приводим кластер к начальному виду как у "х"
        superior=np.reshape(superior,(1,n))
        cluster = np.append(cluster,superior, axis=0)
        list_of_clusters.append(cluster)
        #print(f'After delete {cluster=}')
        x = remove_cluster(x,cluster)
        #print(f'{x=}')
    return list_of_clusters

# def select_clusters(size_pop, n, m, count_of_clusters, uplimit,lowlimit, fofma,
#                     net_structure_index=0):
#     # n - длина вектора переменных
#     # m - число ближайших соседей
#     def add_column(data,x,z):
#         for i in range(np.shape(data)[0]):
#             for j in range(np.shape(data)[1]):
#                 if (j == 0 or j==1):
#                     data[i][j]=x[i][j]
#                 else:
#                     data[i][j]=z[i]
#         return data
#     def give_min_m(m,distanses):
#         distanses = distanses[distanses[:,2].argsort()]
#         return distanses[0:m]
#     def remove_cluster(x,cluster):
#         to_remove = []
#         for i in range(len(cluster)):
#             for j in range(len(x)):
#                 if (np.all(cluster[i]==x[j])):
#                     to_remove.append(j)
#         x=np.delete(x,to_remove,0)
#         return x
#     list_of_clusters = []
#     np.random.seed(10)
#     x=np.random.uniform(uplimit,lowlimit, (size_pop,n))
#     #print(f'{x=}')
#     for i in range(count_of_clusters):
#         z = []
#         for i in range(size_pop):
#             z.append(fofma(x[i], net_structure_index))
#         data = np.ones((len(z),n+1))
#         data = add_column(data,x,z)
#         # for i in range(size_pop):
#         #     for j in range(n+1):
#         #         if (j == 0 or j==1):
#         #             data[i][j]=x[i][j]
#         #         else:
#         #             data[i][j]=z[i]

#         data = data[data[:,2].argsort()]
#         data_set = data.copy()
#         #print(data_set)
#         superior = data_set[0,0:n]
#         #print(f'{superior=}')

#         distanses = np.delete(data_set, 0,0)
#         #print(f'{distanses=}')
#         for i in range(len(distanses)):
#             distanses[i][n] = (np.linalg.norm(superior-distanses[i][0:n]))
#         #print(f'{distanses=}')
#         cluster = give_min_m(m,distanses)
#         #print(f'{cluster=}')
#         cluster = np.delete(cluster, 2,1)       # приводим кластер к начальному виду как у "х"
#         superior=np.reshape(superior,(1,n))
#         cluster = np.append(cluster,superior, axis=0)
#         list_of_clusters.append(cluster)
#         #print(f'After delete {cluster=}')
#         x = remove_cluster(x,cluster)
#         #print(f'{x=}')
#     return list_of_clusters