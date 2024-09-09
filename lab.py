import random

def get_removal_indices_for_larger_list(original_list, required_items):
    return sorted(random.sample(range(len(original_list)), len(original_list)-required_items), reverse=True)

def get_inserted_indices_for_smaller_list(original_list, required_items):
    return sorted(random.sample(range(1, required_items-1), required_items-len(original_list)), reverse=True)

# Example usage
original_list = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10']
required_items = 5

indices_to_remove = get_removal_indices_for_larger_list(original_list, required_items)
print("Indices to remove:", indices_to_remove)



# Example usage
original_list = ['item1', 'item2', 'item3', 'item4', 'item5']
required_items = 7

inserted_indices = get_inserted_indices_for_smaller_list(original_list, required_items)
print("Indices to insert:", inserted_indices)