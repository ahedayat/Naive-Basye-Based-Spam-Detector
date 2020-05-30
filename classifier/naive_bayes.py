
class Naive_Bayes:
    def __init__(self, train_data, parameters, alpha):
        self.parameters = parameters
        self.alpha = alpha
        self.train_data = train_data

    def p_w_spam(self, word):
        if word in self.train_data.columns:
            return (self.train_data.loc[self.train_data['label'] == 'spam', word].sum() + self.alpha) / (self.parameters['nspam'] + self.alpha*self.parameters['nvoc'])
        else:
            return 1

    def p_w_ham(self, word):
        if word in self.train_data.columns:
            return (self.train_data.loc[self.train_data['label'] == 'ham', word].sum() + self.alpha) / (self.parameters['nham'] + self.alpha*self.parameters['nvoc'])
        else:
            return 1

    def classify(self, message):
        p_spam_given_message = self.parameters['pspam']
        p_ham_given_message = self.parameters['pham']
        for word in message:
            p_spam_given_message *= self.p_w_spam(word)
            p_ham_given_message *= self.p_w_ham(word)

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_ham_given_message < p_spam_given_message:
            return 'spam'
        else:
            return 'needs human classification'
