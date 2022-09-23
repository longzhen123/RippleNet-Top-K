import numpy as np
import pandas as pd


def load_kg(data_dir):
    # print(rel_dict)
    # print('load_kg...')
    edges = pd.read_csv(data_dir + 'kg.txt', delimiter='\t', header=None).values
    kg_dict = {}
    relation_set = set()
    entity_set = set()
    for edge in edges:
        head = edge[0]
        tail = edge[1]
        relation = edge[2]

        if head not in kg_dict:
            kg_dict[head] = []

        kg_dict[head].append([relation, tail])

        entity_set.add(head)
        entity_set.add(tail)
        relation_set.add(relation)

    n_entity = len(entity_set)

    return kg_dict, n_entity, len(relation_set)


def data_split(ratings_np, item_set, ratio):
    # print('data split...')

    positive_records = get_records(ratings_np)
    train_set = []
    eval_records = dict()
    test_records = dict()
    for user in positive_records:
        pos_record = positive_records[user]

        size = len(pos_record)
        test_size = int(size * 0.2)
        eval_size = int(size * 0.1)

        test_indices = np.random.choice(size, test_size, replace=False)
        rem_indices = list(set(range(size)) - set(test_indices))
        eval_indices = np.random.choice(rem_indices, eval_size, replace=False)
        train_indices = list(set(rem_indices) - set(eval_indices))

        if ratio < 1:
            size = int(len(train_indices) * ratio)
            if size < 1:
                size = 1
            train_indices = np.random.choice(train_indices, size, replace=False)

        train_set.extend([user, pos_record[i], 1] for i in train_indices)

        neg_items = list(item_set - set(pos_record))
        train_neg_items = np.random.choice(neg_items, len(train_indices), replace=False).tolist()
        rem_neg_items = list(set(neg_items) - set(train_neg_items))
        eval_neg_items = np.random.choice(rem_neg_items, 50, replace=False).tolist()
        test_neg_items = np.random.choice(list(set(rem_neg_items) - set(eval_neg_items)), 50, replace=False).tolist()

        train_set.extend([user, neg_item, 0] for neg_item in train_neg_items)

        if len(eval_indices) != 0:
            eval_records[user] = eval_neg_items + np.random.choice([pos_record[i] for i in eval_indices], 1).tolist()
            test_records[user] = test_neg_items + np.random.choice([pos_record[i] for i in test_indices], 1).tolist()

    return train_set, eval_records, test_records


def load_ratings(data_dir):

    data_np = pd.read_csv(data_dir + 'ratings.txt', delimiter='\t', header=None).values

    return data_np


def get_records(data_set):

    records = dict()

    for pair in data_set:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in records:
                records[user] = []

            records[user].append(item)

    return records


def load_data(args):
    data_dir = './data/' + args.dataset + '/'
    ratings_np = load_ratings(data_dir)
    item_set = set(ratings_np[:, 1])
    user_set = set(ratings_np[:, 0])
    train_set, eval_records, test_records = data_split(ratings_np, item_set, args.ratio)

    kg_dict, n_entity, n_relation = load_kg(data_dir)
    n_entity = n_entity
    n_user = len(user_set)
    n_item = len(item_set)
    data = [n_entity, n_user, n_item, n_relation, train_set, eval_records, test_records, kg_dict]

    return data
