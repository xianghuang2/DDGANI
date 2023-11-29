def get_data_index(data_name, params):
    for index,dict in enumerate(params):
        if dict['name'] == data_name:
            return index
    return -1