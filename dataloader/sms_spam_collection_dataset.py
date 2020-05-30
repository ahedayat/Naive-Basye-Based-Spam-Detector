import pandas as pd


class SMS_SPAM:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_pickle('{}'.format(self.path))
        self.messages = self.get_messages()

    def get_messages(self):
        messages = list()
        for ix, row in self.data.iterrows():
            messages.append((row['label'], row['sms_message']))
        return messages

    def __getitem__(self, ix):
        label, message = self.messages[ix]
        return label, message

    def __len__(self):
        return len(self.data)
