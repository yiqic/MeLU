config = {
    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    # cuda setting
    'use_cuda': False,
    # model setting
    'inner': 1,
    'lr': 5e-3,
    'local_lr': 5e-3,
    'learn_local_lr': True,
    'batch_size': 32,
    'num_epoch': 3,
    # candidate selection
    'num_candidate': 20,
    'include_item_embeddings': True,
    'tests_per_epoch': 80,
    'enable_data_aug': False,
    'train_k': 5,
}

# states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
states = ["warm_state", "user_cold_state"]
