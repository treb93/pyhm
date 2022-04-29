class DataLoader:
    """Data loading, cleaning and pre-processing."""

    def __init__(self, data_paths, fixed_params):
        self.data_paths = data_paths

        (
            self.user_item_train,
            self.user_item_test,
            self.item_sport_interaction,
            self.user_sport_interaction,
            self.sport_sportg_interaction,
            self.item_feat_df,
            self.user_feat_df,
            self.sport_feat_df,
            self.sport_onehot_df,
        ) = format_dfs(
            self.data_paths.train_path,
            self.data_paths.test_path,
            self.data_paths.item_sport_path,
            self.data_paths.user_sport_path,
            self.data_paths.sport_sportg_path,
            self.data_paths.item_feat_path,
            self.data_paths.user_feat_path,
            self.data_paths.sport_feat_path,
            self.data_paths.sport_onehot_path,
            fixed_params.remove,
            fixed_params.ctm_id_type,
            fixed_params.item_id_type,
            fixed_params.weeks_of_purchases,
            fixed_params.days_of_clicks,
            fixed_params.lifespan_of_items,
            fixed_params.report_model_coverage,
        )

        if fixed_params.report_model_coverage:
            print('Reporting model coverage')
            (_, _, _, _, _, _, _, _
             ) = format_dfs(
                self.data_paths.train_path,
                self.data_paths.test_path,
                self.data_paths.item_sport_path,
                self.data_paths.user_sport_path,
                self.data_paths.sport_sportg_path,
                self.data_paths.item_feat_path,
                self.data_paths.user_feat_path,
                self.data_paths.sport_feat_path,
                0,  # remove 0
                fixed_params.ctm_id_type,
                fixed_params.item_id_type,
                fixed_params.weeks_of_purchases,
                fixed_params.days_of_clicks,
                fixed_params.lifespan_of_items,
                fixed_params.report_model_coverage,
            )

        self.ctm_id, self.pdt_id, self.spt_id = create_ids(
            self.user_item_train,
            self.item_sport_interaction,
            self.item_feat_df,
            item_id_type=fixed_params.item_id_type,
            ctm_id_type=fixed_params.ctm_id_type,
            spt_id_type=fixed_params.spt_id_type,
        )

        (
            self.adjacency_dict,
            self.ground_truth_test,
            self.ground_truth_purchase_test,
            self.user_item_train_grouped,
            # Will be grouped if duplicates != 'keep_all'. Used for recency
            # edge feature
        ) = df_to_adjacency_list(
            self.user_item_train,
            self.user_item_test,
            self.item_sport_interaction,
            self.user_sport_interaction,
            self.sport_sportg_interaction,
            self.ctm_id,
            self.pdt_id,
            self.spt_id,
            item_id_type=fixed_params.item_id_type,
            ctm_id_type=fixed_params.ctm_id_type,
            spt_id_type=fixed_params.spt_id_type,
            discern_clicks=fixed_params.discern_clicks,
            duplicates=fixed_params.duplicates,
        )

        self.graph_schema = {('user',
                              'buys',
                              'item'): list(zip(self.adjacency_dict['user_item_src'],
                                                self.adjacency_dict['user_item_dst'])),
                             ('item',
                                 'bought-by',
                                 'user'): list(zip(self.adjacency_dict['user_item_dst'],
                                                   self.adjacency_dict['user_item_src'])),
                             }
