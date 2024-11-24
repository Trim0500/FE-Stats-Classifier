import matplotlib.pyplot as plt
from math import ceil
from numpy import array
from pandas import DataFrame
from torch import tensor, sqrt, sort
from StatsTableSingleton import StatsTableSingleton

class CharacterStatsAnalysis():
    """
        Class that abstracts the statistical and analytical procedures for the character stat line.

        Class Members: 
            stats_names: List representing the stat labels to use in plots
            stats_colors: List representing the stat label colors to use in plots
        
        Instance Members: 
            tensor_device: String representing the hardware device to run tensors on
            character_name: String representing the character name for filtering
            index_labels: List representing the game entries for the character
            stats_table: StatsTableSingleton instance representing the "database" connection to retrieve data
            filtered_stats_dataframe: DataFrame instance representing the complete stat line table for the character
            stats: Tensor instance representing the complete stat line floating point data from the table
            normalized_stats: Tensor instance representing the normalized stat lines
            ave_stats: Tensor instance representing the normalized average stat line
            stat_percentage: Tensor instance representing the percentage distribution of the stat lines by row
        
    """

    stats_names = ["HP","Atk","Skl","Spd","Lck","Def","Res"]
    stats_colors = ["darkorange","darkred","darkblue","darkgreen","gold","#cccc00","royalblue"]

    def __init__(_self, _tensor_device: str, _character_name: str, _index_labels: list) -> None:
        _self.tensor_device = _tensor_device
        _self.character_name = _character_name
        _self.index_labels = _index_labels
        _self.stats_table = StatsTableSingleton()
        _self.filtered_stats_dataframe = DataFrame([])
        _self.stats = tensor([])
        _self.normalized_stats = tensor([])
        _self.ave_stats = tensor([])
        _self.stat_percentage = tensor([])

    def filter_dataframe(_self) -> None:
        """
            Method to use the singleton instance to get the filtered dataframe and convert to tensor.

            Raises:
                exc: Exception
        """
        try:
            _self.filtered_stats_dataframe = _self.stats_table.get_stats_by_name(_self.character_name, _self.index_labels)
            _self.stats = tensor(_self.filtered_stats_dataframe.to_numpy(), dtype=float, device=_self.tensor_device)
        except Exception as exc:
            print(f"[CharacterStatsAnalysis/filter_dataframe]: {str(exc)}")

    def normalize_stats(_self) -> None:
        """
            Method to utilize the stats of a character across game appearences to determine normalized stats.
            Note: Some game appearences don't always have values for specific stats, so mean and standard deviation is affected.

            Raises:
                exc: Exception
        """
        try:
            mask = _self.stats != -99
            mean_tensor = (_self.stats * mask).sum(dim=0) / mask.sum(dim=0)

            std_list = []

            for i in range(len(mean_tensor)):
                stat_tensor = _self.stats[:,i]
                stat_tensor = stat_tensor[stat_tensor != -99]
                std_list.append(sqrt(((stat_tensor - mean_tensor[i])**2).sum() / len(stat_tensor)))

            std_tensor = tensor(std_list, device=_self.tensor_device)
            _self.normalized_stats = _self.stats - mean_tensor / std_tensor
            _self.ave_stats = (_self.normalized_stats * mask).sum(dim=0) / mask.sum(dim=0)
        except Exception as exc:
            print(f"[CharacterStatsAnalysis/normalize_stats]: {str(exc)}")

    def plot_ave_stat_distribution(_self) -> None:
        """
            Method to visualize the normalized average stat line for a particular character through a pie plot.

            Raises:
                exc: Exception
        """
        try:
            plt.figure(figsize=(9,9), facecolor="lightgrey")

            patches, texts, pcts = plt.pie(abs(_self.ave_stats).cpu(),
                                            counterclock=False,
                                            startangle=90,
                                            colors=CharacterStatsAnalysis.stats_colors,
                                            labels=CharacterStatsAnalysis.stats_names,
                                            autopct="%1.1f%%",
                                            textprops={ "size": "larger", "fontweight": "bold" },
                                            wedgeprops={ "linewidth": 1.5, "edgecolor": "white" },
                                            shadow={'ox': -0.02, 'edgecolor': 'none', 'shade': 0.6})
            plt.setp(pcts, color="white")

            plt.title(f"Average Normalized Stat Distribution for {_self.character_name}", fontdict={ "fontsize": "x-large", "fontweight": "bold" }, loc="left")

            plt.show()
        except Exception as exc:
            print(f"[FE_Stats_Data_Analysis/plot_ave_stat_distribution]: {str(exc)}")

    def process_percentage_tensor(_self) -> None:
        """
            Method to utilize the stats of a character across game appearences to determine percentage distribution.
            Note: Some game appearences don't always have values for specific stats, so totals are affected.

            Raises:
                exc: Exception
        """
        try:
            mask = _self.normalized_stats > -90.0
            stat_totals = (_self.normalized_stats * mask).sum(dim=1)

            percs = []

            for i in range(len(_self.normalized_stats)):
                buffer = _self.normalized_stats[i]
                buffer[buffer < -90.0] = 0.0
                percs.append(((abs(buffer) / stat_totals[i]) * 100).tolist())

            _self.stat_percentage = tensor(percs, device=_self.tensor_device)
        except Exception as exc:
            print(f"[FE_Stats_Data_Analysis/process_percentage_tensor]: {str(exc)}")

    def plot_all_percentage_stat_lines(_self) -> None:
        """
            Method to visualize all normalized stat lines for a particular character through a pie plot.

            Raises:
                exc: Exception
        """
        try:
            fig, axes = plt.subplots(nrows=ceil(_self.stat_percentage.shape[0] / 4), ncols=4, figsize=(12,12), layout="constrained", facecolor="lightgrey")

            for i, ax in enumerate(axes.flat):
                if i >= len(_self.stat_percentage):
                    break

                buffer = _self.stat_percentage[i].cpu().to(int)

                name_buffer = array(CharacterStatsAnalysis.stats_names)

                color_buffer = array(CharacterStatsAnalysis.stats_colors)

                if (buffer == 0).sum() > 0:
                    match_indices = (buffer != 0).nonzero().view(-1)
                    buffer = buffer[match_indices.numpy()]

                    name_buffer = name_buffer[match_indices.numpy()]
                    
                    color_buffer = color_buffer[match_indices.numpy()]
                
                buffer, indices = sort(buffer, descending=True)

                patches, texts, pcts = ax.pie(buffer,
                                                counterclock=False,
                                                startangle=90,
                                                colors=color_buffer,
                                                labels=name_buffer,
                                                autopct="%1.1f%%",
                                                textprops={ "size": "small", "fontweight": "bold" },
                                                wedgeprops={ "linewidth": 1.5, "edgecolor": "white" },
                                                shadow={'ox': -0.02, 'edgecolor': 'none', 'shade': 0.6})
                plt.setp(pcts, color="white")

                ax.set_title(f"{_self.character_name} {_self.index_labels[i]} Stat Dist.", fontdict={ "fontsize": "small", "fontweight": "bold" }, loc="left")

            plt.show()
        except Exception as exc:
            print(f"[FE_Stats_Data_Analysis/process_percentage_tensor]: {str(exc)}")

