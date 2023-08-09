from abc import ABC


class Ranker(ABC):
    def rank(self, sents):
        """Describe this.
        
        Parameters
        ----------
        sents : list[String]
            the sentences to be ranked

        Returns
        -------
        Generator[int]
            generates the indices of the selected sentences, in order
        
        """
        pass    

