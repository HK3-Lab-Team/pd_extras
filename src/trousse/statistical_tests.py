import bisect
import itertools
import logging
import os
import traceback
from typing import Tuple

import bokeh.plotting as bk
import numpy as np
import pandas as pd
import scipy.stats as ss

import scikit_posthocs as sp
from medplot.utils.bokeh_boxplot import make_boxplot
from medplot.utils.seaborn_bar_plot import get_show_samples_per_group

from .dataframe_with_info import DataFrameWithInfo

logger = logging.getLogger(__name__)
NA_VALUE = 0
SUBGROUP_PAIR_COMBINATION_COLUMN = "Subgroup_Pair_Combination"


def is_subgroup_populated(
    subgroup, sample_count_threshold_for_statistics, feature_list
):
    # Check if there are enough samples in the subgroup with a Not NaN value  to do statistical tests
    not_nan_count_per_feature = subgroup[feature_list].notna().sum()

    return not_nan_count_per_feature >= sample_count_threshold_for_statistics


def subgroups_statistical_test(
    df_group_slice,
    subgroup_col_name,
    groupby_column,
    separation_count,
    full_plot_list,
    feature_list,
    H_value_by_group_by_subgroup,
    p_value_by_group_by_subgroup,
    sample_count_threshold_for_statistics=20,
    p_value_stat_func=ss.kruskal,
    p_value_thresholds=(0.05, 0.01),
    complete_df_info: DataFrameWithInfo = None,
    show_separated_boxplot=False,
    show_separated_boxplot_params=(30, 0.01),
):
    """
    This function will be applied to a DataFrame compute the p-values between pairs of distributions.
    Particularly it will group the df_info rows based on: their 'groupby_column' and
    a 'subgroup_col_name' column.  Then, for each group of rows, it will consider the values
    they have in every feature and it will consider these values as a distribution that
    can be compared with every other. To be more precise the single distributions will
    be compared pairwise but only when they have the same value of 'groupby_column'
    (and obviously for the same feature). Therefore it will compare the distributions with
    different values of the 'subgroup_col_name' to check their separation.
    Based on the resulting p-values, the function will return a pd.DataFrame with the features
    as columns and the pairwise combinations of the values of the 'subgroup_col_name' column
    repeated for every value of the 'groupby_column'. The p-values will also be encoded based on how
    much separated the partitions are. The p-value intervals will be described by 'p_value_thresholds'.
    @param df_group_slice: pd.DataFrame slice with only rows from a specific group
    @param groupby_column: The rows of df_info will be split according to the values in this column
    @param subgroup_col_name: This is the column whose values will determine the different partitions.
        The function will consider every possible combination of its values and will perform a
        statistical test on each of these pairs
    @param sample_count_threshold_for_statistics: Important parameter that defines how many samples
        are required to have a reliable p-value. The partitions that will not have enough samples,
        will be associated with a value '-1'
    @param p_value_thresholds: These are the thresholds that will be used to give a code to p-values
        (to distinguish meaningful separations). Default value set to (0.05, 0.01) -> p-value intervals will be
        '0' -> p-value = 1 - 0.05
        '1' -> p-value =  0.05 - 0.01
        '2' -> p-value =  0.01 - 0
    @param complete_df_info: DataFrameWithInfo instance with the full DataFrame where we will find data
        to draw boxplot from. Required only if 'show_separated_boxplot' == True
    @param show_separated_boxplot: Bool: Option if you want to show the boxplots with many separated
        distributions. Each boxplot will contain distributions from a specific breed only and a
        specific feature. Each distribution is characterized by a different value of the 'subgroup_col_name'
    @param show_separated_boxplot_params: Tuple to decide how much the partitions must be separated in order to
        show the boxplot. The first element is an integer to describe how many p-values have to be separated
        in order to show the corresponding boxplot. The second is the p-value threshold to decide that
        a separation exists
    @param p_value_stat_func: This is the function that can be used for calculating p-value. Default value set to
        scipy_stats.kruskal
    @param feature_list: Optional argument if you want to analyze only few specific features/columns
    @return: pd.DataFrame with p-values for every possible combination of two partitions. Based on
        'p_value_thresholds' argument, it contains:
            '-1' -> if not enough samples in one of the two compared partitions
            '0' -> No separation (p-value from '1' to the first value of 'p-value thresholds')
            '1' -> Little separation (p-value from the first value of 'p-value thresholds' to the second
                value of 'p-value thresholds')
            '2' -> Higher separation ...
    """
    # Group by according to the 'subgroup_col_name'
    group_name = df_group_slice.name
    df_groupby_subgroups = df_group_slice.groupby(subgroup_col_name)

    if complete_df_info is None and show_separated_boxplot:
        logging.error(
            "If you want to see the most separated boxplot, you must provide the original "
            "full DataFrameWithInfo instance"
        )
    # Initialize variables
    features_already_plotted = set()
    full_plot_list[group_name] = []
    partial_plot_list = []

    # STEP 1. Drop the subgroups/keys which contain less than 'sample_count_threshold_for_statistics' values.
    # This first check prevents useless computations later.
    is_subgroup_populated_bool_map = df_groupby_subgroups.apply(
        is_subgroup_populated,
        sample_count_threshold_for_statistics=sample_count_threshold_for_statistics,
        feature_list=feature_list,
    )
    # STEP 2. Prepare the map with stastical tests result for every possible combination.
    #         For each breed and combination of 2 possible subgroups we say in which
    #         interval the p-value is, based on "p_value_thresholds" argument for each feature.
    #         Initialize every value to NA_VALUE (-1) like every test was not possible.
    #         We will change that for the ones we manage to test.

    # Column List of the DF containing p-values
    col_list = [groupby_column, SUBGROUP_PAIR_COMBINATION_COLUMN]
    col_list.extend(feature_list)

    # Create the list of possible combinations of the subgroups (extract from np array)
    unique_subgroup_values = sorted(
        [x[0] for x in df_groupby_subgroups[subgroup_col_name].unique()]
    )
    possible_combination_subgroups = list(
        itertools.combinations(unique_subgroup_values, 2)
    )
    possible_combination_subgroups.sort()

    # Create an array with the group_name and every possible subgroup combination
    p_value_index_array = np.array(
        [
            [group_name] * len(possible_combination_subgroups),
            possible_combination_subgroups,
        ]
    ).transpose()
    # Create zero array for p_value related to every feature
    p_value_feat_array = np.full(
        shape=(len(possible_combination_subgroups), len(feature_list)),
        fill_value=-1,
        dtype=int,
    )
    # Concatenate the two parts adding new columns
    p_value_array = np.concatenate((p_value_index_array, p_value_feat_array), axis=1)
    p_value_separation_map = pd.DataFrame(p_value_array, columns=col_list)

    # STEP 3. Fill up the DF with the p-values we can calculate with statistical tests

    # Prepare the dictionaries to store p, H actual values for every pair of distributions (with enough samples only!)
    (
        H_value_by_group_by_subgroup[group_name],
        p_value_by_group_by_subgroup[group_name],
    ) = ({}, {})

    groupby_subgroups_ids = df_groupby_subgroups.groups

    logging.info(f"Analyzing feature p-values from {group_name}")

    # df_correct.df[feature_list].apply(
    #     get_statistical_test_per_feature,
    #     groups_breed_subgr_ids=groupby_subgroups_ids,
    #     is_subgroup_populated_bool_map=is_subgroup_populated_bool_map,
    #     features_already_plotted=features_already_plotted,
    #     possible_combination_subgroups=possible_combination_subgroups,
    #     breed_name=group_name,
    #     p_value_separation_map=p_value_separation_map,
    #     subgroup_col_name=subgroup_col_name, separation_count=separation_count,
    #     full_plot_list=full_plot_list,
    #     H_value_by_breed_by_subgroup=H_value_by_group_by_subgroup,
    #     p_value_by_breed_by_subgroup=p_value_by_group_by_subgroup,
    #     p_value_stat_func=p_value_stat_func,
    #     p_value_thresholds=p_value_thresholds, complete_df_info=complete_df_info,
    #     show_separated_boxplot=show_separated_boxplot,
    #     show_separated_boxplot_params=show_separated_boxplot_params,
    #     axis=0
    # )
    for feat in feature_list:

        feat_separation_counter = 0
        # Loop over every possible combination of the populated subgroups only
        for g1_id, g2_id in possible_combination_subgroups:

            # Combination of 2 subgroups
            pair = (g1_id, g2_id)
            # Initialize the dictionaries for every pair
            (
                H_value_by_group_by_subgroup[group_name][pair],
                p_value_by_group_by_subgroup[group_name][pair],
            ) = ({}, {})

            # Check if the two subgroups have enough samples to perform a statistical test
            #   If there are no groups with that id, it means there are no samples at all
            try:
                is_subgroup_populated_bool = (
                    is_subgroup_populated_bool_map.at[g1_id, feat]
                    and is_subgroup_populated_bool_map.at[g2_id, feat]
                )
            except KeyError:
                is_subgroup_populated_bool = False

            if is_subgroup_populated_bool:

                # Select the 2 distributions corresponding to the selected (group, subgroup, feature)
                # correct combination
                group_1_distribution = (
                    complete_df_info.df.loc[groupby_subgroups_ids[g1_id]][feat]
                    .dropna()
                    .values
                )
                group_2_distribution = (
                    complete_df_info.df.loc[groupby_subgroups_ids[g2_id]][feat]
                    .dropna()
                    .values
                )

                # RUN THE STATISTICAL TESTS
                # "try" needed because the two pair_data distributions may contain only identical
                # values -> ValueError from 'p_value_stat_func' method (since it is non-uniform)
                try:
                    # Check which function has been called since they accept different arguments
                    if ss.kruskal:
                        kruskal_result = p_value_stat_func(
                            group_1_distribution, group_2_distribution
                        )
                        p_value_by_group_by_subgroup[group_name][pair][
                            feat
                        ] = kruskal_result.pvalue
                    elif sp.posthoc_nemenyi:
                        pair_data = [group_1_distribution, group_2_distribution]
                        (
                            H_value_by_group_by_subgroup[group_name][pair][feat],
                            p_value_by_group_by_subgroup[group_name][pair][feat],
                        ) = p_value_stat_func(pair_data).at[2, 1]
                    else:
                        raise NotImplementedError(
                            f"The function {p_value_stat_func} is not implemented yet. The "
                            f"two possible functions are 'ss.kruskal', 'sp.posthoc_nemenyi'"
                        )

                except ValueError as e:
                    if "identical" in str(e):
                        # Identical values cannot be analyzed (keep going)
                        logging.info(
                            f"Every value of the subgroups {g1_id} and {g2_id} for feature {feat} "
                            "in {breed_name} is identical. The feature contains many identical values and "
                            "may not be really meaningful. Skipping this (-1 is assigned)"
                        )
                        continue
                    else:
                        # Something unexpected happened (break)
                        traceback.print_exc()
                        break
                # Based on the p-value we assign an index according to the interval it belongs to,
                # defined by the list of
                #  thresholds 'p_value_thresholds'. We use binary search to find the right interval and
                #  we give 0/1/2/3.. code (-1 is for NA values)
                p_value_interval_label = len(p_value_thresholds) - bisect.bisect_right(
                    p_value_thresholds,
                    p_value_by_group_by_subgroup[group_name][pair][feat],
                )
                # Increase the counter of the corresponding p_value bin
                separation_count[p_value_interval_label] += 1
                # Fill the corresponding cell with p_value_interval_label
                p_value_separation_map.loc[
                    p_value_separation_map[SUBGROUP_PAIR_COMBINATION_COLUMN] == pair,
                    feat,
                ] = p_value_interval_label

                if (
                    p_value_by_group_by_subgroup[group_name][pair][feat]
                    < show_separated_boxplot_params[1]
                ):
                    feat_separation_counter += 1

                # If the partitions/pairs of this feature are usually separated (meanly there are more than
                # 'show_separated_boxplot_params[0]' high p-values, add the 12 bw-plot to the final plot
                # to manually check the separation
                if (
                    show_separated_boxplot
                    and (feat_separation_counter > show_separated_boxplot_params[0])
                    and (feat not in features_already_plotted)
                ):

                    # Create a new plot for every level
                    p, _ = make_boxplot(
                        complete_df_info,
                        selected_group_to_plot=(groupby_column, group_name),
                        level_col_name=subgroup_col_name,
                        level_id=3,
                        input_feat_x=feat,
                        remove_outliers=True,
                        show_outliers=False,
                    )
                    # Add to the set, so it won't be re-plotted for the same breed and feature
                    features_already_plotted.add(feat)
                    # Append the figure to the list, so we can make a grid of plots
                    partial_plot_list.append(p)

                    # After 30 plots, store the partial list and start a new one to prevent a performance decrease
                    if len(partial_plot_list) == 19:
                        print("Reached 30 plots!")
                        # Append a partial list of boxplots
                        full_plot_list[group_name].append(partial_plot_list)
                        # Reset the partial plot list
                        partial_plot_list = []

    # STEP 5. Append the remaining List of plots for this breed
    full_plot_list[group_name].append(partial_plot_list)

    logging.info(f"Done elaborating {group_name}. Overall found : {separation_count}")

    return p_value_separation_map


def draw_and_save_plot_list(plot_list, file_name=None, column_num_in_grid=3):
    # make a grid
    grid = bk.gridplot(
        plot_list, ncols=column_num_in_grid, plot_width=500, plot_height=500
    )

    if file_name is None:
        # Just show the plots
        bk.show(grid)
    else:
        bk.show(grid)
        bk.save(grid, filename=os.path.join(file_name), title=file_name)


def draw_every_plot_from_apply(full_plot_list):
    for breed in full_plot_list.keys():
        for part_plot_list_id in range(len(full_plot_list[breed])):
            draw_and_save_plot_list(
                full_plot_list[breed][part_plot_list_id],
                file_name=os.path.join(
                    f"{breed}_Kruskal_most_separated_classes_{part_plot_list_id}.html"
                ),
            )


def compute_separation_per_breed_per_subgroup_vs_feature(
    df_info: DataFrameWithInfo,
    groupby_column: str,
    subgroup_col_name: str,
    sample_count_threshold_for_statistics: int,
    p_value_thresholds: Tuple = (0.05, 0.01),
    first_groups_sorted=None,
    first_group_elements_threshold=10,
    p_value_stat_func=ss.kruskal,
    feature_list=None,
    show_separated_boxplot=False,
    show_separated_boxplot_params=(30, 0.01),
    filename=None,
):
    """
    This function will compute the p-values between pairs of distributions.
    Particularly it will group the df_info rows based on: their 'groupby_column' and
    a 'subgroup_col_name' column.  Then, for each group of rows, it will consider the values
    they have in every feature and it will consider these values as a distribution that
    can be compared with every other. To be more precise the single distributions will
    be compared pairwise but only when they have the same value of 'groupby_column'
    (and obviously for the same feature). Therefore it will compare the distributions with
    different values of the 'subgroup_col_name' to check their separation.
    Based on the resulting p-values, the function will return a pd.DataFrame with the features
    as columns and the pairwise combinations of the values of the 'subgroup_col_name' column
    repeated for every value of the 'groupby_column'. The p-values will also be encoded based on how
    much separated the partitions are. The p-value intervals will be described by 'p_value_thresholds'.
    @param df_info: DataFrameWithInfo instance with data
    @param groupby_column: The rows of df_info will be split according to the values in this column
    @param first_groups_sorted: Optionally, the user may directly input which values from the
        'groupby_column' should be analyzed
    @param first_group_elements_threshold: If the possible values in 'groupby_column' are too
        many, it could be better choosing only the most populated ones. This integer value selects how many
        most populated groups should be considered. Default value set to 10
    @param subgroup_col_name: This is the column whose values will determine the different partitions.
        The function will consider every possible combination of its values and will perform a
        statistical test on each of these pairs
    @param sample_count_threshold_for_statistics: Important parameter that defines how many samples
        are required to have a reliable p-value. The partitions that will not have enough samples,
        will be associated with a value '-1'
    @param p_value_thresholds: These are the thresholds that will be used to give a code to p-values
        (to distinguish meaningful separations). Default value set to (0.05, 0.01) -> p-value intervals will be
        '0' -> p-value = 1 - 0.05
        '1' -> p-value =  0.05 - 0.01
        '2' -> p-value =  0.01 - 0
    @param show_separated_boxplot: Bool: Option if you want to show the boxplots with many separated
        distributions. This is ot verify if a very strong separation really exist by manually checking it.
        Each boxplot will contain distributions from a specific breed only and a specific feature.
        Each distribution is characterized by a different value of the 'subgroup_col_name'
    @param show_separated_boxplot_params: Tuple to decide how much the partitions must be separated in order to
        show the boxplot. The first element is an integer to describe how many p-values have to be separated
        in order to show the corresponding boxplot. The second is the p-value threshold to decide that
        a separation exists
    @param p_value_stat_func: This is the function that can be used for calculating p-value. Default value set to
        scipy_stats.kruskal
    @param feature_list: Optional argument if you want to analyze only few specific features/columns
    @param filename: If this argument is provided, the plots will be saved there, otherwise they won't be saved at all.
    @return: pd.DataFrame with p-values for every possible combination of two partitions. Based on
        'p_value_thresholds' argument, it contains:
            '-1' -> if not enough samples in one of the two compared partitions
            '0' -> No separation (p-value from '1' to the first value of 'p-value thresholds')
            '1' -> Little separation (p-value from the first value of 'p-value thresholds' to the second
                value of 'p-value thresholds')
            '2' -> Higher separation ...
    """
    # If feature_list argument is not provided, we use all the valid columns from df
    if feature_list is None:
        feature_list = df_info.med_exam_col_list

    # STEP 1. If we don't know the names of the N most populated groups, we calculate them
    if first_groups_sorted is None:
        # Count samples per group
        df_samples_per_group = get_show_samples_per_group(
            df_info, groupby_column, show_plot=False
        )
        # Select only first elements
        first_groups_sorted = list(
            df_samples_per_group.loc[:first_group_elements_threshold][groupby_column]
        )

    # Select only the top populated groups based on first_groups_sorted
    df_most_populated_groups = df_info.df[
        df_info.df[groupby_column].isin(first_groups_sorted)
    ]
    # Create groups
    df_groupby_slice = df_most_populated_groups.groupby(groupby_column)

    # STEP 2. INITIALIZATION before apply
    full_plot_list = {}
    H_value_by_group_by_subgroup, p_value_by_group_by_subgroup = {}, {}
    # Dict to count the separations per threshold
    separation_count = {}
    for t in range(len(p_value_thresholds) + 1):
        separation_count[t] = 0
    p_value_thresholds = sorted(
        p_value_thresholds, reverse=False
    )  # Possible p-value thresholds

    # STEP 3. Function applied to every group
    p_value_separation_map = df_groupby_slice.apply(
        subgroups_statistical_test,
        H_value_by_group_by_subgroup=H_value_by_group_by_subgroup,
        p_value_by_group_by_subgroup=p_value_by_group_by_subgroup,
        subgroup_col_name=subgroup_col_name,
        feature_list=feature_list,
        groupby_column=groupby_column,
        separation_count=separation_count,
        complete_df_info=df_info,
        sample_count_threshold_for_statistics=sample_count_threshold_for_statistics,
        full_plot_list=full_plot_list,
        p_value_stat_func=p_value_stat_func,
        show_separated_boxplot=show_separated_boxplot,
        show_separated_boxplot_params=show_separated_boxplot_params,
        p_value_thresholds=p_value_thresholds,
    )

    logging.info(f"Found these values based on thresholds: {separation_count}")

    if show_separated_boxplot:
        logging.info(
            f"Plotting the boxplots for the top {first_group_elements_threshold} most "
            "populated groups of column {groupby_column}"
        )
        draw_every_plot_from_apply(full_plot_list)

    # Reset the index to 'groupby_column' and 'SUBGROUP_PAIR_COMBINATION_COLUMN'
    p_value_separation_map = p_value_separation_map.reset_index(drop=True)
    p_value_separation_map = p_value_separation_map.set_index(
        [groupby_column, SUBGROUP_PAIR_COMBINATION_COLUMN]
    )

    return p_value_separation_map


if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path

    sys.path.append("..")
    # try:
    from trousse.utils.dataframe_with_info import (
        DataFrameWithInfo,
        import_df_with_info_from_file,
    )

    # except ImportError:
    #     from bwplot.smvet_utils.utils.dataframe_with_info import DataFrameWithInfo

    logging.basicConfig(
        format="%(asctime)s \t %(levelname)s \t Module: %(module)s \t %(message)s ",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )

    CWD = Path(os.path.abspath(os.path.dirname("__file__"))).parents[0]  # Repo DIR
    DF_PATH = CWD / "data" / "output_data" / "df_patient"
    df_info = import_df_with_info_from_file(str(DF_PATH))
    df_info.metadata_as_features = False
    # Comparing only young patients (first two ids)
    df_info.df = df_info.df[df_info.df["AGE_bin_id"].isin(["2", "1"])]

    feature_list = df_info.med_exam_col_list

    p_value_separation_map = compute_separation_per_breed_per_subgroup_vs_feature(
        df_info=df_info,
        groupby_column="GROUPS",  # trivial column
        subgroup_col_name="AGE_bin_id",
        first_group_elements_threshold=10,
        feature_list=feature_list,
        sample_count_threshold_for_statistics=10,
        p_value_thresholds=[0.05] + list(np.logspace(-10, -100, num=10, base=10)),
        p_value_stat_func=ss.kruskal,  # sp.posthoc_nemenyi,
        show_separated_boxplot=False,
        # show_separated_boxplot_params=(30, 0.01),
        # first_groups_sorted=[0, ],
        # filename='age_comparison_per_feat_p_values.png'
    )
    p_value_age_per_feature = p_value_separation_map.iloc[0]
    p_value_age_sort = np.sort(p_value_age_per_feature.values)
    p_value_age_names = p_value_age_per_feature.index.values[
        np.argsort(p_value_age_per_feature.values)
    ][::-1]

    print(p_value_age_names[:23])
    separated_feat = [
        "percentage_vessel_right",
        "vessel_volumes_mean",
        "ggo_volumes_mean",
        "percentage_ggo_right",
        "original_glszm_SmallAreaLowGrayLevelEmphasis",
        "original_glszm_LowGrayLevelZoneEmphasis",
    ]

    # for feat in separated_feat:
    #     bwplot = make_boxplot(df_info=db_join_hu_med_rad,
    #                           level_col_name='ground_glass',
    #                           input_feat_x=feat,
    #                           remove_outliers=False, hover_tool_list=('study_uid',),
    #                           show_outliers=False, return_outliers=False,
    #                           plot_title=f"{feat}_value_per_GGO",
    #                           circle_fill_transparency=0.9)
    #     bk.output_file(f'{feat}_distr_per_ggo')
    #     export_png(bwplot, filename=f'{feat}_distr_per_ggo.png')
    #     bk.save(bwplot)
    #     bk.show(bwplot)
    print("the end")
