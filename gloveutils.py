"""
    Glove Handler toolbox (for my mini playground setup)
    Glove data :
        https://nlp.stanford.edu/projects/glove/
"""
import numpy as np 

try:
    get_ipython()  # type: ignore # This will raise an exception in non-interactive mode
    from tqdm.autonotebook import tqdm, trange
except NameError:
    from tqdm import tqdm
    
SIZE_50 = "50"
SIZE_100 = "100"
SIZE_200 = "200"
SIZE_300 = "300"
glove_path = "glove.6B/glove.6B.{item}d.txt"

class GloveBox:
    """
    GloveBox class is a utility for handling GloVe word embeddings. It supports loading
    word embeddings from GloVe files for the experiments I do in this respiratory.

    Attributes:
        path (str): The file path for the GloVe embeddings.
        search_history (dict): A cache of previously searched words and their embeddings.
        missing_words (list): A list of words that could not be found in the GloVe dataset.
        shelf (dict): A dictionary containing the loaded word embeddings from GloVe.
    """
    def __init__(self, size: str = "50"):
        """
        Initialize the GloveBox with a specified embedding size.
        
        Args:
            size (str): The size of the GloVe embeddings to load ('50', '100', '200', '300').
        """
        self.path = glove_path.format(item=size)
        self.search_history = {}
        self.missing_words = []
        self.shelf = {}

    def enter_basement(self):
        """
        Load GloVe embeddings from the specified file into memory.
        """
        print(f"Entering basement now. Fetching books . . .")
        with open(self.path, encoding='utf-8') as file:
            for line in tqdm(file):
                values = line.split()
                word = values[0]
                coefs = np.fromiter(map(float, values[1:]), dtype=np.float32)
                self.shelf[word] = coefs
        print(f"N.o of books loaded in shelf! There are {len(self.shelf)} books on the shelves")
    
    def search(self, search_word: str):
        """
        Search for the embedding of a given word in the GloVe dataset.
        
        Args:
            search_word (str): The word to search for in the GloVe embeddings.
        
        Returns:
            np.ndarray or str: The embedding of the word if found, or a message if not found.
        """
        if search_word in self.search_history:
            return self.search_history[search_word]
        
        self.search_history[search_word] = 0
        for key_word in self.shelf:
            if key_word == search_word:
                found = self.shelf[search_word]
                assert found.shape == (50,), f"Unusual loaded size {found.shape} while trying to find word {search_word}"
                found = found.reshape(1, -1)
                self.search_history[search_word] = found
                return found
        
        self.missing_words += [search_word]
        return f"Could not find the word: {search_word}"
    
    def unbox(self, list_items: list):
        """
        Retrieve embeddings for a list of words.
        
        Args:
            list_items (list): A list of words to retrieve embeddings for.
        
        Returns:
            np.ndarray: A stacked array of word embeddings.
        """
        n = len(list_items)
        list_embeds = []
        for word in list_items:
            assert isinstance(word, str), f'Expect strings but received {type(word)}'
            word_embedding = self.search(word.lower())
            
            if isinstance(word_embedding, np.ndarray):
                list_embeds += [word_embedding]
                
        embedding = np.stack(list_embeds)
        assert embedding.shape[0] == n, f"Unusual embedding size: {embedding.shape}"
        return embedding 
    
if __name__ == '__main__':
    # how to use me
    book = GloveBox()
    book.enter_basement()
    print(book.search('tom'))