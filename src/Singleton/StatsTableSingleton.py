from pandas import DataFrame, read_csv
from typing import Any

class StatsTableSingletonMeta(type):
    instance = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        try:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwds)
                cls._instances[cls] = instance
            
            return cls._instances[cls]
        except AttributeError:
            cls._instances = {}
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance

            return cls._instances[cls]
    
class StatsTableSingleton(metaclass=StatsTableSingletonMeta):
    """
        Class that abstracts the "database" connection through the singleton pattern.\n
        Points to the `FE_Stats_Data.csv` file located in the `Data` directory.
    """

    def get_stats_by_name(_self, _character_name: str, _index_labels: list) -> DataFrame:
        """
            Method to slice the dataframe based on character name.
            Will also rename the indices from the index label list argument.

            Args:
                _character_name: String that represents the target character's name to act as a key for filtering.add()
                _index_labels: List of labels to replace the resulting dataframe's index labels.

            Returns:
                filtered_stat_lines_dataframe: A new DataFrame instance representing the filtered stat line dataframe for the target character.

            Raises:
                exc: Exception
        """
        stat_lines_dataframe = read_csv("Data/FE_Stats_Data.csv")
        # [FE Stats Classifier/FE_Stats_Data_Analysis]: If NaN is found, means we shouldn't include that rows value in calculations later. 
        # So replace with a sentinel value.
        stat_lines_dataframe.fillna(-99, inplace=True)
        # [FE Stats Classifier/FE_Stats_Data_Analysis]: Ensure all numeric columns are ints for consistency
        stat_lines_dataframe = stat_lines_dataframe.astype({ "skl": int, "lck": int })
        filtered_stat_lines_dataframe = stat_lines_dataframe[stat_lines_dataframe.character.__eq__(_character_name)].reset_index()
        # [FE Stats Classifier/FE_Stats_Data_Analysis]: Drop last column as it doesn't matter here
        # [FE Stats Classifier/FE_Stats_Data_Analysis]: No need for the character column after filter
        filtered_stat_lines_dataframe = filtered_stat_lines_dataframe.drop(["index","character","label"], axis=1)

        index_dict = {}

        for i in filtered_stat_lines_dataframe.index:
            index_dict[i] = _index_labels[i]

        filtered_stat_lines_dataframe = filtered_stat_lines_dataframe.rename(index=index_dict)

        return filtered_stat_lines_dataframe
