import pandas as pd
from pre_processing import normalize_text, no_marks, diction_nag_pos_not


class DataSource(object):
    def _load_raw_data(self, filename, is_train=True):
        a = []
        b = []
        regex = 'train_'
        if not is_train:
            regex = 'test_'
        with open(filename, 'r', encoding='utf8') as file:
            for line in file:
                if regex in line:
                    b.append(a)
                    a = [line]
                elif line != '\n':
                    a.append(line)
        b.append(a)
        return b[1:]

    def _create_row(self, sample, is_train=True):
        d = {}
        d['id'] = sample[0].replace('\n', '')
        review = ""
        if is_train:
            for clause in sample[1:-1]:
                review += clause.replace('\n', '').strip()
            d['label'] = int(sample[-1].replace('\n', ''))
        else:
            for clause in sample[1:]:
                review += clause.replace('\n', '').strip()
        d['review'] = review
        return d

    def load_data(self, filename, is_train=True):

        raw_data = self._load_raw_data(filename, is_train)
        lst = []

        for row in raw_data:
            lst.append(self._create_row(row, is_train))

        return lst

    def transform_to_dataset(self, x_set, y_set):
        X, y = [], []
        for document, topic in zip(list(x_set), list(y_set)):
            document = normalize_text(document)
            X.append(document.strip())
            y.append(topic)
            # Augmentation bằng cách remove dấu tiếng Việt
            X.append(no_marks(document))
            y.append(topic)
        return X, y

    def return_data(self, file_name):
        ds = DataSource()
        data = pd.DataFrame(ds.load_data(file_name))
        new_data = []
        # Thêm mẫu bằng cách lấy trong từ điển Sentiment (nag/pos)
        nag_list, pos_list, not_list = diction_nag_pos_not()
        for index, row in enumerate(pos_list):
            new_data.append(['pos' + str(index), '0', row])
        for index, row in enumerate(nag_list):
            new_data.append(['nag' + str(index), '1', row])

        new_data = pd.DataFrame(new_data, columns=list(['id', 'label', 'review']))
        data = data.append(new_data, ignore_index=True)
        return data