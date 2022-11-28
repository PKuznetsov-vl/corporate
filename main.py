# import autotime
import math
import time
from collections import defaultdict
from collections import deque

import numpy as np
import pandas as pd

all_vertices_in_components_of_strong_connectivity = []
T_list = []
data_core_graph = []

SH_MODE = 1  # only SH nodes are considered final holders
LEVELS = 20  # this parameter should be equal or exceed the number of "onion layers" in data. 20 is a bit of overkill for safety


# todo ref that part
def load_data(path):
    data = pd.read_csv(path, usecols={'founder_inn', 'inn', 'capital_p'},
                       dtype={'founder_inn': str, 'inn': str, 'capital_p': float})

    data = data.rename(columns={'founder_inn': 'participant_id',
                                'inn': 'organisation_inn',
                                'capital_p': 'equity_share'})
    data = data[(data['equity_share'] > 0) &
                (data.participant_id != data.organisation_inn)]
    # data.head()
    data_orig = data.copy()

    return data


def get_serial_number_by_vertice(v):
    for i in range(len(data_core_graph)):
        if (data_core_graph.index[i] == v):
            return data_core_graph['serial_number'][i]


def from_edges(edges):
    # translate list of edges to list of nodes

    class Node:
        def __init__(self):
            # root is one of:
            # None: not yet visited
            # -1: already processed
            # non-negative integer: lowlink
            self.root = None
            self.successors = []

    nodes = defaultdict(Node)
    for v, w in edges:
        nodes[v].successors.append(nodes[w])

    for i, v in nodes.items():  # name the nodes for final output
        v.id = i

    return nodes.values()


def tarjan(V):
    def strongconnect(v, S):
        v.root = pos = len(S)
        S.append(v)

        for w in v.successors:
            if w.root is None:  # not yet visited
                yield from strongconnect(w, S)

            if w.root >= 0:  # still on stack
                v.root = min(v.root, w.root)

        if v.root == pos:  # v is the root, return everything above
            res, S[pos:] = S[pos:], []
            for w in res:
                w.root = -1
            yield [r.id for r in res]

    for v in V:
        if v.root is None:
            yield from strongconnect(v, [])


def find_stationary(A):
    # note: the matrix is row stochastic.
    # A markov chain transition will correspond to left multiplying by a row vector.
    # We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(A.T)
    evec1 = evecs[:, np.isclose(evals, 1)]

    # Since np.isclose will return an array, we've indexed with an array
    # so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()

    # eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    return stationary


def get_vertice_by_serial_number(serial_number):
    return (data_core_graph[data_core_graph['serial_number'] == serial_number].index[0])


def get_A_for_component(component):
    vertices_one_component = {}
    tmp = {'participant_id': [], 'organisation_inn': [], 'equity_share': [], 'serial_number': []}
    cur_data_core_graph = pd.DataFrame(tmp).set_index('participant_id')

    A_cur_matrix = np.zeros((len(component), len(component)), dtype=float)

    for i in range(len(component)):
        vertices_one_component[i] = component[i]
        all_vertices_in_components_of_strong_connectivity.append(get_vertice_by_serial_number(component[i]))

    cur_data_core_graph = pd.concat(
        [cur_data_core_graph, data_core_graph[data_core_graph.serial_number.isin(component)]])
    component_inns = set(cur_data_core_graph.index.to_numpy())
    cur_data_core_graph = cur_data_core_graph[cur_data_core_graph.organisation_inn.isin(component_inns)]
    for i in range(len(cur_data_core_graph)):
        a = get_key_by_value(vertices_one_component, get_serial_number_by_vertice(cur_data_core_graph.index[i]))
        b = get_key_by_value(vertices_one_component,
                             get_serial_number_by_vertice(cur_data_core_graph['organisation_inn'][i]))
        A_cur_matrix[a][b] = cur_data_core_graph['equity_share'][i]

    return A_cur_matrix


def get_key_by_value(dictionary, value):
    for k, v in dictionary.items():
        if v == value:
            return k


# предрасчет
# def dataop():
#     path = './founder_sort.csv'
#
#     data = pd.read_csv(path, usecols={'founder_inn', 'inn', 'capital_p'},
#                        dtype={'founder_inn': str, 'inn': str, 'capital_p': float})
#
#     # data.shape
#
#     data = data.rename(columns={"founder_inn": "organisation_inn",
#                                 'inn': 'participant_id',
#                                 'capital_p': 'equity_share'})
#     data = data[(data['equity_share'] > 0) &
#                 (data.participant_id != data.organisation_inn)]
#     data.head()
#     gdata = data.groupby('organisation_inn').sum().reset_index()
#     dict_companies = dict(gdata.values)
#     data['equity_share'] = data['equity_share'] / np.array([dict_companies[num] for num in data['organisation_inn']])
#
#     data['super_holder'] = ~pd.Series(data.participant_id).isin(data.organisation_inn)
#     data['super_target'] = ~pd.Series(data.organisation_inn).isin(data.participant_id)
#
#     return data


def warm_up(data):
    print('pruning SH and ST nodes of the external layer...')
    rdata = data.loc[(data['super_holder'] == False) & (data['super_target'] == False)]
    edges_left = len(rdata)
    print('layer 1:', edges_left, 'edges left')

    crdata = rdata.copy()

    for i in range(1, LEVELS):
        is_sh = ~pd.Series(rdata.participant_id).isin(rdata.organisation_inn)
        current_unique_sh = pd.Series(rdata.loc[is_sh == True].participant_id.value_counts().keys())
        print('current SH', len(current_unique_sh))
        curr_sh_prop_name = 'is_level_' + str(i) + '_SH'
        crdata[curr_sh_prop_name] = pd.Series(crdata.participant_id).isin(current_unique_sh)

        is_st = ~pd.Series(rdata.organisation_inn).isin(rdata.participant_id)
        current_unique_st = pd.Series(rdata.loc[is_st == True].organisation_inn.value_counts().keys())
        print('current ST', len(current_unique_st))
        curr_st_prop_name = 'is_level_' + str(i) + '_ST'
        crdata[curr_st_prop_name] = pd.Series(crdata.organisation_inn).isin(current_unique_st)

        rdata = rdata.loc[(is_sh == False) & (is_st == False)]
        print('layer {}: {} edges left'.format(i + 1, len(rdata)))

    data_core = rdata.copy()
    core_holders = rdata.participant_id.value_counts()
    print('unique core holders:', len(core_holders))
    core_targets = rdata.organisation_inn.value_counts()
    print('unique core targets:', len(core_targets))

    data_core_graph = data_core[['organisation_inn', 'equity_share']]
    data_core_graph.index = data_core['participant_id']
    # we will determine the serial number later
    data_core_graph.insert(2, "serial_number", 0)
    print(data_core_graph)

    double_edges = {}

    i = 0
    while (len(data_core_graph) > 0 and data_core_graph.index.value_counts()[i] > 1):
        double_edges[i] = (data_core_graph.index.value_counts().index[i])
        i += 1

    print(double_edges)

    current_serial_number = 0
    index_serial_number = 0

    for i in range(len(data_core_graph)):
        if (data_core_graph.index[i] in double_edges.values()):
            # if repeated edge
            current_serial_number = get_key_by_value(double_edges, data_core_graph.index[i])
        else:
            while (index_serial_number in double_edges):
                index_serial_number += 1
            current_serial_number = index_serial_number
            index_serial_number += 1

        data_core_graph.loc[data_core_graph.index[i], 'serial_number'] = current_serial_number
    print('gr', data_core_graph.head())

    table_of_edges = []
    for i in range(len(data_core_graph)):
        table_of_edges.append((get_serial_number_by_vertice(data_core_graph.index[i]),
                               get_serial_number_by_vertice(data_core_graph['organisation_inn'][i])))
    print('Total edges: ', len(table_of_edges))

    for component in tarjan(from_edges(table_of_edges)):
        if (len(component) < 2):
            continue
        D = np.zeros((len(component), len(component)), dtype=float)
        A_matrix_one_component = get_A_for_component(component)
        correct_component = True

        for i in range(len(A_matrix_one_component)):
            s = 0
            for j in range(len(A_matrix_one_component)):
                s += A_matrix_one_component[j][i]

            if (not math.isclose(1.0, s)):
                correct_component = False

        component_vertices = []
        if (correct_component):
            T_current = np.zeros((len(component), len(component)), dtype=float)

            stationary = find_stationary(A_matrix_one_component)

            for m in range(len(component)):
                component_vertices.append(get_vertice_by_serial_number(component[m]))
                for n in range(len(component)):
                    T_current[m][n] = stationary[m]
            T_list.append([component_vertices, T_current])
        else:
            B = np.eye(len(A_matrix_one_component)) - A_matrix_one_component
            try:
                T_current = A_matrix_one_component.dot(np.linalg.inv(B))
            except np.linalg.LinAlgError as err:
                print(err)
                print("component = ", component)
                print("A = ", A_matrix_one_component)
                print("B = ", B)
                continue
            for m in range(len(T_current)):
                component_vertices.append(get_vertice_by_serial_number(component[m]))
                cur_col_sum = 0

                for n in range(len(T_current)):
                    cur_col_sum += T_current[n][m]

                if (cur_col_sum == 0):
                    print("component = ", component)
                else:
                    for n in range(len(T_current)):
                        T_current[n][m] = T_current[n][m] / cur_col_sum

            T_list.append([component_vertices, T_current])


suitable_vertices = dict()
stack_vertices = []
used_ancestors = set()


def find_Z_by_company_inn(company_inn):
    for i in range(len(T_list)):
        for j in range(len(T_list[i][0])):
            if (T_list[i][0][j] == company_inn):
                return T_list[i], j


def set_suitable_vertices(company_inn: str, data: pd.DataFrame) -> bool:
    additional_graph = data.copy(deep=True)
    additional_graph.rename(columns={'super_holder': 'terminality'}, inplace=True)
    additional_graph = additional_graph.drop(columns=['super_target'], axis=1)
    additional_graph = additional_graph.set_index('organisation_inn')
    st_time = time.monotonic()
    suitable_vertices.clear()
    stack_vertices.clear()
    used_ancestors.clear()
    if (len(data[data['organisation_inn'] == company_inn]) == 0 and not stack_vertices):
        print("Without owners")
        return False

    while (True):
        if (time.monotonic() - st_time > 60.0):
            print("Вычисление хэш-таблицы более 60 секунд...")
            return False
        if (len(data[data.organisation_inn == company_inn]) == 0):
            suitable_vertices[company_inn] = [0]  # for terminal vertices, the initial weight is 0
            if (len(stack_vertices) == 0):
                return True
            company_inn = stack_vertices.pop()
            used_ancestors.add(company_inn)
            continue

        cur_Z = []
        is_component_ancestors_need_added = True
        is_company_in_strong_component = (company_inn in all_vertices_in_components_of_strong_connectivity)

        # if the company is in a strongly connected component,
        # then we will find the ownership component, write it in 'cur_Z'
        # no further actions will happen with 'cur_Z' if the company is not in the component

        if (is_company_in_strong_component):
            cur_Z, index_cur_company_in_Z = find_Z_by_company_inn(company_inn)

        cur_data = additional_graph[additional_graph.index == company_inn]
        noncomponent_ancestors = [0]
        component_ancestors = [0]
        ancestors = cur_data['participant_id'].values

        for ancestor in ancestors:
            if (ancestor not in used_ancestors):
                stack_vertices.append(ancestor)
            cur_ancestor_inn = cur_data[cur_data['participant_id'] == ancestor].participant_id[0]
            if (cur_ancestor_inn == None):
                print("Error related to 'cur_ancestor_inn'")
                return False
            cur_ancestor_share = cur_data[cur_data['participant_id'] == ancestor].equity_share[0]
            if (is_company_in_strong_component):
                if (ancestor in all_vertices_in_components_of_strong_connectivity):
                    # cur_company in component and ancestor in component
                    if (is_component_ancestors_need_added):
                        for j in range(len(cur_Z[0])):
                            cur_ancestor_inn = cur_Z[0][j]
                            cur_ancestor_share = cur_Z[1][j][index_cur_company_in_Z]
                            component_ancestors.append([cur_ancestor_inn + 'L', cur_ancestor_share])
                        is_component_ancestors_need_added = False
                else:
                    # cur_company in component, ancestor not in component
                    noncomponent_ancestors.append([cur_ancestor_inn, cur_ancestor_share])
            else:
                if (ancestor in all_vertices_in_components_of_strong_connectivity):
                    # cur_company not in component, ancestor in component
                    noncomponent_ancestors.append([cur_ancestor_inn + 'R', cur_ancestor_share])
                else:
                    # cur_company not in component, ancestor not in component
                    noncomponent_ancestors.append([cur_ancestor_inn, cur_ancestor_share])

        if (is_company_in_strong_component):
            suitable_vertices[company_inn + 'R'] = component_ancestors
            suitable_vertices[company_inn + 'L'] = noncomponent_ancestors
        else:
            suitable_vertices[company_inn] = noncomponent_ancestors

        if (stack_vertices):
            company_inn = stack_vertices.pop()
            used_ancestors.add(company_inn)
        else:
            return True


# requested_company = "4104002024"
# set_suitable_vertices(requested_company)


def set_additional_vertex():
    global suitable_vertices
    suitable_vertices_copy = suitable_vertices.copy()
    for k, v in suitable_vertices.items():
        if (k[len(k) - 1] == 'L'):
            s = 0
            for i in range(1, len(v)):
                s += v[i][1]
            if ((not math.isclose(s, 1.0)) & (not math.isclose(s, 0))):
                suitable_vertices_copy[k[:len(k) - 1]] = [1]
                tmp = v.copy()
                tmp.append([k[:len(k) - 1], round(1 - round(s, 7), 7)])
                suitable_vertices_copy[k] = tmp
    suitable_vertices = suitable_vertices_copy


def get_vertex_without_RL(company_inn: str) -> str:
    if (company_inn[len(company_inn) - 1] == 'R' or company_inn[len(company_inn) - 1] == 'L'):
        return str(company_inn[:len(company_inn) - 1])
    else:
        return company_inn


def set_terminality_to_table(company_inn: str):
    global suitable_vertices
    suitable_vertices_copy = suitable_vertices.copy()
    for k, v in suitable_vertices.items():
        s = 0
        for i in range(1, len(v)):
            s += v[i][1]
        if (s == 0.0 and (k[len(k) - 1] != 'R')):
            v.append('T')
            v[0] = 0
        else:
            v.append('N')
            if (get_vertex_without_RL(k) == company_inn and (k[len(k) - 1] != 'L')):
                v[0] = 1
            else:
                v[0] = 0
        # print(k, " ", v)
    suitable_vertices = suitable_vertices_copy


# set_terminality_to_table(requested_company)


def get_vertex_name_by_presence(company_inn: str) -> str:
    if (suitable_vertices.get(company_inn + 'R') == None):
        # if not in the component
        if (suitable_vertices.get(company_inn) != None):
            # if not in the component, but there is such a vertex
            return company_inn
        else:
            return ""
    else:
        # if in the component, then we will start the crawl from the RIGHT lobe
        return company_inn + 'R'


def get_equity_share(company_inn: str, data: pd.DataFrame, data_orig: pd.DataFrame):
    intermediaries_owners = dict()
    queue_vertices = deque()
    final_owners = dict()
    final_owners_lst = []
    intermediaries_owners_lst = []
    st_time = time.monotonic()
    queue_vertices.clear()
    used_vertices = set()
    queue_vertices.append(get_vertex_name_by_presence(company_inn))
    s = 0
    while (queue_vertices):
        if (time.monotonic() - st_time > 60.0):
            print("Вычисление конечных владельцев более 60 секунд...")
            return False
        cur_company = queue_vertices.pop()
        company_info_dict = suitable_vertices.get(cur_company)

        if (company_info_dict == None):
            print("Couldn't find the entered company")
            if company_inn in data.participant_id.values:
                print('Found Person')
                return find_dec(data, company_inn)
            else:
                return False

        for i in range(1, len(company_info_dict) - 1):
            cur_owner = company_info_dict[i][0]
            cur_equity_share = company_info_dict[i][1]
            owner_info_dict = suitable_vertices.get(cur_owner)
            if (owner_info_dict == None):
                print(f"An error occurred with the owner search {cur_owner}")
                return False

            owner_info_dict[0] = company_info_dict[0] * cur_equity_share
            if (owner_info_dict[len(owner_info_dict) - 1] == "T"):
                s += owner_info_dict[0]

                cur_owner = get_vertex_without_RL(cur_owner)
                if (cur_owner in final_owners):
                    # if we considered the path to this vertex, then we sum it up with the current weight
                    final_owners.update({cur_owner: owner_info_dict[0] + final_owners.get(cur_owner)})
                else:
                    # if you met for the first time, then we set the ownership share
                    final_owners[cur_owner] = owner_info_dict[0]
            else:  # вероятно ошибка здесь
                cur_owner = get_vertex_without_RL(cur_owner)
                if (cur_owner in intermediaries_owners):
                    # if we considered the path to this vertex, then we sum it up with the current weight
                    intermediaries_owners.update({cur_owner: owner_info_dict[0] + intermediaries_owners.get(cur_owner)})
                else:
                    # if you met for the first time, then we set the ownership share
                    intermediaries_owners[cur_owner] = owner_info_dict[0]
                if (cur_owner not in used_vertices):
                    queue_vertices.append(cur_owner)
                    used_vertices.add(cur_owner)
    # presentation from the most important owners to the smaller ones
    list_final_owners = list(final_owners.items())
    list_intermediaries_owners = list(intermediaries_owners.items())
    list_final_owners.sort(key=lambda i: i[1])
    list_final_owners.reverse()
    list_intermediaries_owners.sort(key=lambda i: i[1])
    list_intermediaries_owners.reverse()
    owner_counter = 1
    print(f"Ownership share in the company {company_inn}:")
    print('s', s)
    # Зачем нам эта нормаировка если  аглоритм и так нормирует?
    norm_coef = 1 / s  # какой-нибудь try catch при s = 0

    print("Final owners:")
    for owner in list_final_owners:
        print(f'{owner_counter}. {owner[0]} = {(owner[1] * norm_coef * 100):.4f}%')
        owner_counter += 1
        final_owners_lst.append([owner[0], round(owner[1] * norm_coef * 100, 2)])

    print("Intermediaries owners:")
    owner_counter = 1
    for owner in list_intermediaries_owners:
        print(f'{owner_counter}. {owner[0]} = {(owner[1] * norm_coef * 100):.4f}%')
        owner_counter += 1
        intermediaries_owners_lst.append([owner[0], round(owner[1] * norm_coef * 100, 2)])
    dec = find_dec(df_f=data, inn=company_inn)

    for el in dec:
        el[1] = round(el[1] * 100, 2)
    # new_dec.append((el[0],round( el[1] * norm_coef * 100,2)))
    dec1 = find_dec(df_f=data_orig, inn=company_inn)

    print(f"The total amount of final ownership share is equal to {(s * 100 * norm_coef):.6}%")
    print('parents', find_par(data_orig, company_inn))
    print('dec', dec)
    print('childrens ', dec1)
    return final_owners_lst, find_par(data_orig, company_inn), dec, \
           find_dec(data_orig, company_inn), intermediaries_owners_lst


def find_dec(df_f, inn):
    columns = ['inn', 'childrens']

    # inn =503802414742 # 10000246917 5038107129 7606080127
    df = df_f.loc[df_f['participant_id'] == inn]
    df = df.drop_duplicates('organisation_inn')

    print(df.head())
    if not df.empty:
        return list(map(list, zip(df.organisation_inn, df.equity_share)))
    else:
        return []


def find_par(df_f, inn):
    columns = ['inn', 'parents']

    # inn =503802414742 # 10000246917 5038107129 7606080127
    df = df_f.loc[df_f['organisation_inn'] == inn]
    df = df.drop_duplicates('participant_id')

    if not df.empty:
        print(df.head())
        return list(map(list, zip(df.participant_id, df.equity_share)))
    else:
        return []


def get_csv():
    requested_company = '9728040365'
    set_suitable_vertices(requested_company)
    set_additional_vertex()
    set_terminality_to_table(requested_company)
    final_owners_lst, parents_lst, dec, childrens, intermediaries_owners_lst \
        = get_equity_share(requested_company)
    final_owners_lst = [{'inn': el[0], 'ownership_p': el[1], 'name': 'get_name_db(el[0])', 'status': 'ascendents'} for
                        el
                        in final_owners_lst]
    parents_lst = [{'inn': el[0], 'ownership_p': el[1], 'name': 'get_name_db(el[0])', 'status': 'parents'} for el in
                   parents_lst]
    dec = [{'inn': el[0], 'ownership_p': el[1], 'name': 'get_name_db(el[0])', 'status': 'descendents'} for el in dec]
    childrens = [{'inn': el[0], 'ownership_p': el[1], 'name': 'get_name_db(el[0])', 'status': 'childrens'} for el in
                 childrens]
    intermediaries_owners_lst = [
        {'inn': el[0], 'ownership_p': el[1], 'name': 'get_name_db(el[0])', 'status': 'intermediaries_owners'}
        for el in intermediaries_owners_lst]
    # df = pd.concat([pd.DataFrame(final_owners_lst),pd.DataFrame(parents_lst),pd.DataFrame(dec),
    #                 pd.DataFrame(childrens),pd.DataFrame(intermediaries_owners_lst)])
    df = pd.concat([pd.DataFrame(final_owners_lst), pd.DataFrame(childrens)])
    print(df.head(10))


# get_csv()


def get_equity_share_old(company_inn: str) -> bool:
    queue_vertices = deque()
    final_owners = dict()
    st_time = time.monotonic()
    queue_vertices.clear()
    final_owners.clear()
    queue_vertices.append(get_vertex_name_by_presence(company_inn))
    s = 0
    while (queue_vertices):
        if (time.monotonic() - st_time > 60.0):
            print("Вычисление конечных владельцев более 60 секунд...")
            return False
        cur_company = queue_vertices.pop()
        company_info_dict = suitable_vertices.get(cur_company)

        if (company_info_dict == None):
            print("Couldn't find the entered company")
            return False

        # company_info_dict[0] = vertex weight (ownership share)
        # company_info_dict[1...len(company_info_dict)-2] = (owner, equity_share)
        # company_info_dict[len(company_info_dict)-1] = terminality
        for i in range(1, len(company_info_dict) - 1):
            cur_owner = company_info_dict[i][0]
            cur_equity_share = company_info_dict[i][1]
            owner_info_dict = suitable_vertices.get(cur_owner)
            if (owner_info_dict == None):
                print(f"An error occurred with the owner search {cur_owner}")
                return False

            owner_info_dict[0] = company_info_dict[0] * cur_equity_share
            if (owner_info_dict[len(owner_info_dict) - 1] == "T"):
                s += owner_info_dict[0]
                cur_owner = get_vertex_without_RL(cur_owner)
                if (cur_owner in final_owners):
                    # if we considered the path to this vertex, then we sum it up with the current weight
                    final_owners.update({cur_owner: owner_info_dict[0] + final_owners.get(cur_owner)})
                else:
                    # if you met for the first time, then we set the ownership share
                    final_owners[cur_owner] = owner_info_dict[0]
            else:
                queue_vertices.append(cur_owner)
    # presentation from the most important owners to the smaller ones
    list_owners = list(final_owners.items())
    list_owners.sort(key=lambda i: i[1])
    list_owners.reverse()

    print(f"Ownership share in the company {company_inn}:")
    for owner_counter, owner in enumerate(list_owners):
        print(f'{owner_counter}. {owner[0]} = {(owner[1] * 100 * (1.0 / s)):.4f}%')

    print(f"The total amount of ownership share is equal to {(s * 100 * (1.0 / s)):.9}%")
    return True


def without_terminal_owners():
    # старая версия алгоритма
    path = './founder_sort.csv'
    data = load_data(path)
    data_orig = data.copy()
    dict_companies = dict(data.groupby('organisation_inn').sum().reset_index().values)
    data['equity_share'] = data['equity_share'] / np.array([dict_companies[num] for num in data['organisation_inn']])

    data['super_holder'] = ~pd.Series(data.participant_id).isin(data.organisation_inn)
    data['super_target'] = ~pd.Series(data.organisation_inn).isin(data.participant_id)

    warm_up(data=data)

    requested_company = '9728040365'  # "503802414742" 352806209266 2304071215
    set_suitable_vertices(company_inn=requested_company, data=data)
    set_additional_vertex()
    set_terminality_to_table(company_inn=requested_company)
    get_equity_share_old(requested_company)


def with_terminal_owners():
    path = './founder_sort.csv'
    data = load_data(path)
    data_orig = data.copy()
    dict_companies = dict(data.groupby('organisation_inn').sum().reset_index().values)
    data['equity_share'] = data['equity_share'] / np.array([dict_companies[num] for num in data['organisation_inn']])
    data['super_holder'] = ~pd.Series(data.participant_id).isin(data.organisation_inn)
    data['super_target'] = ~pd.Series(data.organisation_inn).isin(data.participant_id)

    warm_up(data=data)

    requested_company = '9728040365'  # "503802414742" 352806209266 2304071215
    set_suitable_vertices(company_inn=requested_company, data=data)
    set_additional_vertex()
    set_terminality_to_table(company_inn=requested_company)
    get_equity_share(company_inn=requested_company, data=data, data_orig=data_orig)


if __name__ == '__main__':
    with_terminal_owners()
# without_terminal_owners()
