import collections

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ...pd_extras.dataframe_with_info import FeatureOperation
from ...pd_extras.feature_enum import OperationTypeEnum


def prepare_featureoperation_combos():
    """
    Create different pairs of FeatureOperation instances to be compared for tests.
    """
    attribute_possible_values = {
        "original_columns": [
            ("original_column_1", "original_column_2"),
            ("original_column_3", "original_column_4"),
        ],
        "derived_columns": ["derived_column_1", "derived_column_2"],
        "encoder": [OneHotEncoder, OrdinalEncoder],
    }
    truth_table_combos_per_attribute = [
        (None, None, True),
        (0, None, True),
        (None, 0, True),
        (0, 0, True),
        (0, 1, False),
    ]
    # For each attribute, create the possible pairs of values based
    # on ``truth_table_combos_per_attribute``
    pairs_per_attribute_with_label = collections.defaultdict(list)
    for attribute, attr_values in attribute_possible_values.items():
        for id_1, id_2, is_attr_equal in truth_table_combos_per_attribute:
            pairs_per_attribute_with_label[attribute].append(
                (
                    None if id_1 is None else attr_values[id_1],
                    None if id_2 is None else attr_values[id_2],
                    is_attr_equal,
                )
            )
    operation_types = [op.value for op in OperationTypeEnum]
    operation_type_pairs_with_labels = [
        (operation_types[0], operation_types[1], False),
        (operation_types[0], operation_types[0], True),
    ]
    # Combine every possible pair of values for each attribute
    feat_operations_pairs_with_labels = []
    for op_type_1, op_type_2, is_op_type_equal in operation_type_pairs_with_labels:
        for (
            orig_column_1,
            orig_column_2,
            is_orig_column_equal,
        ) in pairs_per_attribute_with_label["original_columns"]:
            for (
                derived_column_1,
                derived_column_2,
                is_derived_column_equal,
            ) in pairs_per_attribute_with_label["derived_columns"]:
                for (
                    encoder_1,
                    encoder_2,
                    is_encoder_equal,
                ) in pairs_per_attribute_with_label["encoder"]:
                    is_equal_label = (
                        is_op_type_equal
                        and is_orig_column_equal
                        and is_derived_column_equal
                        and is_encoder_equal
                    )
                    feat_operations_pairs_with_labels.append(
                        (
                            dict(
                                operation_type=op_type_1,
                                original_columns=orig_column_1,
                                derived_columns=derived_column_1,
                                encoder=encoder_1,
                            ),
                            dict(
                                operation_type=op_type_2,
                                original_columns=orig_column_2,
                                derived_columns=derived_column_2,
                                encoder=encoder_2,
                            ),
                            is_equal_label,
                        )
                    )
    return feat_operations_pairs_with_labels
