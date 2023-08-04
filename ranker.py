from abc import ABC


class Ranker(ABC):
    def rank(self, sents):
        """Describe this.
        
        Parameters
        ----------

        Returns
        -------
        
        """
        raise NotImplementedError("implement me!")        

