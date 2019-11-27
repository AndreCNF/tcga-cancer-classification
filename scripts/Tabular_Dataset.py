from torch.utils.data import Dataset

class Tabular_Dataset(Dataset):
    def __init__(self, arr, df, label_name=None):
        # Counter that indicates in which column we're in when searching for the label column
        col_num = 0
        for col in df.columns:
            if 'label' in col or col == label_name:
                # Column number corresponding to the label
                self.label_column = col_num
                break
            col_num += 1
        # Column numbers corresponding to the features
        self.features_columns = list(range(self.label_column)) + list(range(self.label_column + 1, arr.shape[1]))
        # Features
        self.X = arr[:, self.features_columns]
        # Labels
        self.y = arr[:, self.label_column]

    def __getitem__(self, item):
        x_i = self.X[item]
        y_i = self.y[item]
        return x_i, y_i

    def __len__(self):
        return len(self.X)
