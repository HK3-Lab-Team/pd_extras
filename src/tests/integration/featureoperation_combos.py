import collections
import itertools

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ...pd_extras.dataframe_with_info import FeatureOperation
from ...pd_extras.feature_enum import OperationTypeEnum


def eq_featureoperation_fixture():
    """
    Create pairs of FeatureOperation instances to be compared for tests.

    Many different pairs of FeatureOperation instances are created in order to test
    the correct comparison between equal and unequal pairs.
    When comparing FeatureOperation instances, ``__eq__`` method checks the attributes
    related to:
     1. operation type
     2. original_columns (only if they are both not None)
     3. derived_columns (only if they are both not None)
     4. encoder (only if they are both not None)
    Based on this, the method creates the possible pairs (combining same, different
    and None values for each attribute)

    Returns
    -------
    List[Tuple[Dict, Dict, boolean]]
        List of tuples, representing the different FeatureOperation pairs to be tested.
        Therefore each tuple contains: one dict for the attributes of the first
        FeatureOperation instance, a second dict for the attributes of the second
        FeatureOperation instance, and a boolean indicating if the two resulting
        instances are actually equal or not.
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
    # Combine every possible pair of values for each attribute and create a list with
    # these pairs
    feat_operations_pairs_with_labels = []
    for (
        op_type_tuple,
        orig_column_tuple,
        deriv_column_tuple,
        encoder_tuple,
    ) in itertools.product(
        operation_type_pairs_with_labels,
        pairs_per_attribute_with_label["original_columns"],
        pairs_per_attribute_with_label["derived_columns"],
        pairs_per_attribute_with_label["encoder"],
    ):
        # Unpack tuples from itertools
        op_type_1, op_type_2, is_op_type_equal = op_type_tuple
        orig_column_1, orig_column_2, is_orig_column_equal, = orig_column_tuple
        deriv_column_1, deriv_column_2, is_deriv_column_equal, = deriv_column_tuple
        encoder_1, encoder_2, is_encoder_equal = encoder_tuple

        is_equal_label = (
            is_op_type_equal
            and is_orig_column_equal
            and is_deriv_column_equal
            and is_encoder_equal
        )
        feat_operations_pairs_with_labels.append(
            (
                dict(
                    operation_type=op_type_1,
                    original_columns=orig_column_1,
                    derived_columns=deriv_column_1,
                    encoder=encoder_1,
                ),
                dict(
                    operation_type=op_type_2,
                    original_columns=orig_column_2,
                    derived_columns=deriv_column_2,
                    encoder=encoder_2,
                ),
                is_equal_label,
            )
        )

    return feat_operations_pairs_with_labels
