from copy import deepcopy


class AttributeGroup:
    """
    Represents a group of attributes within a dataset, facilitating operations such as filtering and sampling
    based on attribute values.

    Attributes:
        attribute_names (list of str): Names of the attributes that define the group.
                             Example: ["Age_binary"] or ["Age_binary", "sex"]
        attribute_group (tuple): The specific values of attributes that define this group.
        group_df (pd.DataFrame): DataFrame containing only the rows that match the attribute group.
        remaining_df (pd.DataFrame): A copy of `group_df` used for operations like sampling without replacement.
    """

    def __init__(self, attribute_names, attribute_group, df):
        """
        Initializes the AttributeGroup with specified attributes and a DataFrame.

         Args:
             attribute_names (list of str): Names of the attributes that define the group.
             attribute_group (tuple): Specific values of attributes defining the group.
             df (pd.DataFrame): DataFrame from which the group DataFrame is created.
        """
        self.attribute_names = attribute_names
        self.attribute_group = attribute_group
        self.group_df = None
        self.compute_group_df(df)
        self.remaining_df = deepcopy(self.group_df)

    def __str__(self):
        return f"Attributes: {self.attribute_group}, Group Size: {len(self.group_df)}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        """
        Returns the length of the `group_df`.

        Returns:
            The length of the `group_df`.
        """
        return len(self.group_df)

    def __lt__(self, other):
        """
        Compares the length of the `group_df` to the length of another `AttributeGroup` object.

        Args:
            other: Another `AttributeGroup` object.

        Returns:
            True if the length of the `group_df` is less than the length of the `group_df` of the other object, False otherwise.
        """
        return len(self.group_df) < len(other.group_df)

    def compute_group_df(self, df):
        """
        Filters the DataFrame to include only the rows that match the specified attributes of the group.

        Args:
            df (pd.DataFrame): The original DataFrame from which to filter rows.

        Returns:
            None: The result is stored in `group_df`.
        """
        df_mask = None
        for i, name in enumerate(self.attribute_names):
            if i == 0:
                df_mask = df[name] == self.attribute_group[i]
            else:
                df_mask = df_mask & (df[name] == self.attribute_group[i])
        self.group_df = df[df_mask]

    def sample(self, count, replace=False):
        """
        Samples a specified number of items from `remaining_df`, optionally without replacement.

        Args:
            count (int): Number of items to sample.
            replace (bool): If False, sampled items are removed from `remaining_df` to prevent re-sampling.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled items.

        Raises:
            ValueError: If `count` exceeds the number of available items and `replace` is False.
        """
        try:
            sampled_items = self.remaining_df.sample(n=count)
        except:
            if count > len(self.remaining_df):
                print(
                    f"Reducing the size of sampled count from {count} to {len(self.remaining_df)}"
                )
                count = len(self.remaining_df)
                sampled_items = self.remaining_df.sample(n=count)
        if not replace:
            self.remaining_df = self.remaining_df.drop(sampled_items.index)

        return sampled_items
