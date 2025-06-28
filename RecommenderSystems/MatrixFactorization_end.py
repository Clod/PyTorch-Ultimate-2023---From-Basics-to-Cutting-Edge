#%%
import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error
#%% data import
df = pd.read_csv("ratings.csv")
df.head(2)
#%%
print(f"Unique Users: {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")

#%% Data Class
class MovieDataset(Dataset):
    """
    PyTorch Dataset for loading movie ratings data.

    This dataset class is designed to work with the PyTorch DataLoader. It
    takes arrays of user IDs, movie IDs, and their corresponding ratings,
    and provides a way to access them as tensors for model training.

    Args:
        users (np.ndarray): An array of user IDs.
        movies (np.ndarray): An array of movie IDs.
        ratings (np.ndarray): An array of ratings corresponding to each
            user-movie pair.
    """
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    # len(movie_dataset)
    def __len__(self):
        """Returns the total number of ratings in the dataset."""
        return len(self.users)
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
            containing the user ID (long), movie ID (long), and rating (float).
        """
        users = self.users[idx] 
        movies = self.movies[idx]
        ratings = self.ratings[idx]
        
        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(movies, dtype=torch.long),
            torch.tensor(ratings, dtype=torch.float32),
        )
       
#%% Model Class
class RecSysModel(nn.Module):
    """
    A Neural Collaborative Filtering model for predicting movie ratings.
    
    1) Input: The actual inputs to the forward method are simple integer IDs for users 
    and movies (e.g., user_id=5, movie_id=101). These are just pointers.
    2) The "Target" of Learning: The nn.Embedding layers are essentially lookup tables. 
    These tables are initialized with random vectors. During training, the model's entire 
    goal is to adjust the numbers inside these vectors. The backpropagation process, driven 
    by the difference between the predicted rating and the true rating, directly modifies 
    the embedding vectors.
    3) The Goal: The network's purpose is to create meaningful embeddings. A successful 
    training run results in a user_embed table where users with similar tastes have similar 
    vectors, and a movie_embed table where similar movies have similar vectors.

    This model implements a form of matrix factorization using neural networks.
    It learns dense embedding vectors (latent factors) for each user and movie.
    To predict a rating for a given user-movie pair, it looks up their
    respective embeddings, concatenates them, and passes the resulting vector
    through a linear layer to produce a single rating value.
    As you can see, in this recommender system, unlike older LLMs the embeddings are 
    not a pre-processed input; they are the learnable parameters that the network 
    is designed to optimize.

    Args:
        n_users (int): The total number of unique users in the dataset.
        n_movies (int): The total number of unique movies in the dataset.
        n_embeddings (int, optional): The size of the embedding vectors for
            both users and movies. Defaults to 32.
    """
    def __init__(self, n_users, n_movies, n_embeddings = 32):
        super().__init__()        
        self.user_embed = nn.Embedding(n_users, n_embeddings)
        self.movie_embed = nn.Embedding(n_movies, n_embeddings)
        self.out = nn.Linear(n_embeddings * 2, 1)

    def forward(self, users, movies):
        """
        Performs the forward pass to predict ratings for a batch of
        user-movie pairs.

        Args:
            users (torch.Tensor): A tensor of user IDs, shape `(batch_size,)`.
            movies (torch.Tensor): A tensor of movie IDs, shape `(batch_size,)`.

        Returns:
            torch.Tensor: A tensor of predicted ratings, shape `(batch_size, 1)`.
        """
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        x = torch.cat([user_embeds, movie_embeds], dim=1)     
        x = self.out(x)       
        return x

#%% encode user and movie id to start from 0 
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.userId = lbl_user.fit_transform(df.userId.values)
df.movieId = lbl_movie.fit_transform(df.movieId.values)
#%% create train test split
df_train, df_test = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.rating.values
)
#%% Dataset Instances
train_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values
)

valid_dataset = MovieDataset(
    users=df_test.userId.values,
    movies=df_test.movieId.values,
    ratings=df_test.rating.values
)
#%% Data Loaders
# -- Step 6: Create DataLoaders --
# Now that we have our `Dataset` objects, we'll wrap them in `DataLoader`s.
# The DataLoader is a PyTorch utility that makes it incredibly easy and efficient to iterate over our dataset in batches.

# We define the batch size. This is a hyperparameter that determines how many samples the model processes at once before updating its weights.
# A batch size of 4 is very small and used here for demonstration. In a real-world scenario, you'd use a larger size like 64, 128, or 256
# to take better advantage of GPU parallelization and get a more stable gradient estimate.
BATCH_SIZE = 4

# We create the DataLoader for our training set.
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True # This is a crucial parameter for training! `shuffle=True` ensures that the data is randomly shuffled at the beginning of every epoch.
                                       # This prevents the model from learning any spurious patterns related to the order of the data and helps it generalize better.
                          ) 

# We do the same for our test (validation) set.
test_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True # For a final, one-time evaluation, shuffling is not strictly necessary. However, if you are using this loader for validation
                                       # during training (i.e., checking performance after each epoch), shuffling can give a less biased estimate of the validation loss.
                          ) 
#%% Model Instance, Optimizer, and Loss Function
# -- Step 7: Initialize the Model, Optimizer, and Loss Function --

# We create an instance of our RecSysModel.
model = RecSysModel(
    n_users=len(lbl_user.classes_), # The number of unique users, which we get from our fitted LabelEncoder. This sets the size of the user embedding table.
    n_movies=len(lbl_movie.classes_)) # The number of unique movies, from the movie LabelEncoder. This sets the size of the movie embedding table.

# We choose our optimizer. The Adam optimizer is a very popular and effective choice that adapts the learning rate for each parameter.
# We pass `model.parameters()` to tell the optimizer which tensors (weights and biases) it is responsible for updating.
optimizer = torch.optim.Adam(model.parameters())
# We define our loss function. Since we are predicting a continuous value (the rating), this is a regression problem.
# Mean Squared Error (MSE) is the standard loss function for regression. It measures the average squared difference between the true and predicted ratings.
criterion = nn.MSELoss()

#%% Model Training
# -- Step 8: Train the Model --

# We define the number of times we want to iterate over the entire training dataset.
NUM_EPOCHS = 1

# We set the model to training mode. This is important because some layers, like Dropout or BatchNorm, behave differently
# during training and evaluation. `model.train()` tells them to be in "training" mode.
model.train() 
# This is our main training loop. It will run for the specified number of epochs.
for epoch_i in range(NUM_EPOCHS):
    # The `train_loader` is an iterator. In each step of this inner loop, it provides a new batch of data.
    # It conveniently unpacks the tuple we defined in `MovieDataset.__getitem__` into three tensors.
    for users, movies, ratings in train_loader:
        # It's essential to zero out the gradients before each batch. If we didn't, gradients would accumulate
        # from previous batches, leading to incorrect updates.
        optimizer.zero_grad()
        # This is the forward pass. We feed the batch of user and movie IDs to the model to get our predictions.
        y_pred = model(users, movies)         
        # The model outputs a tensor of shape (batch_size, 1), but our `ratings` tensor is shape (batch_size,).
        # We use `unsqueeze(1)` to add a dimension, making its shape (batch_size, 1) so it matches the prediction's shape for the loss calculation.
        y_true = ratings.unsqueeze(dim=1)
        # We calculate the loss by comparing the model's predictions (`y_pred`) with the actual ratings (`y_true`).
        loss = criterion(y_pred, y_true)
        # This is the magic of PyTorch's autograd system! `loss.backward()` computes the gradient of the loss
        # with respect to all model parameters (our embeddings and linear layer weights).
        loss.backward()
        # `optimizer.step()` uses the computed gradients to update the model's parameters, moving them in the
        # direction that will reduce the loss.
        optimizer.step()

#%% Model Evaluation 
# -- Step 9: Evaluate the Model with Mean Squared Error --

# We'll store all our predictions and true values in these lists to calculate the overall MSE.
y_preds = []
y_trues = []

# We set the model to evaluation mode using `model.eval()`. This is the counterpart to `model.train()` and ensures
# layers like Dropout are turned off, giving deterministic output.
model.eval()
# We wrap our evaluation loop in `with torch.no_grad()`. This tells PyTorch not to calculate gradients,
# which makes the process much faster and uses less memory, as we don't need gradients for inference.
with torch.no_grad():
    # We iterate through our test data loader, which provides batches of unseen data.
    for users, movies, ratings in test_loader: 
        # We get the true ratings. `.detach()` creates a new tensor that is detached from the computation graph (no gradients).
        # `.numpy()` converts it to a NumPy array, and `.tolist()` converts that to a standard Python list.
        y_true = ratings.detach().numpy().tolist()
        # We get the model's predictions. `.squeeze()` removes the extra dimension (from shape (batch_size, 1) to (batch_size,)),
        # and then we convert it to a list just like the true ratings.
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        # We append the batch results to our master lists.
        y_trues.append(y_true)
        y_preds.append(y_pred)

# Now that we have all the predictions and true values, we can use scikit-learn's `mean_squared_error` function to get our final score.
mse = mean_squared_error(y_trues, y_preds)
# We print the result. A lower MSE indicates a better model.
print(f"Mean Squared Error: {mse}")

#%% Users and Items
# -- Step 10: Calculate Precision and Recall @ K --
# MSE is a good overall metric, but for recommender systems, we often care more about ranking metrics like Precision and Recall.
# Here, we'll calculate these metrics for our test set predictions.

# We use a `defaultdict(list)` to create a dictionary where each user ID maps to a list of their (predicted_rating, true_rating) pairs.
# This is a convenient way to group all ratings by user.
user_movie_test = defaultdict(list)
 
# Again, we use `torch.no_grad()` for efficiency as we are only doing inference.
with torch.no_grad():
    # We loop through the test data one more time.
    for users, movies, ratings in test_loader:         
        # Get the model's predictions for the current batch.
        y_pred = model(users, movies)
        # Now we process each sample within the batch individually.
        for i in range(len(users)):
            # `.item()` is a handy method to extract the single scalar value from a tensor that has only one element.
            user_id = users[i].item()
            movie_id = movies[i].item() 
            pred_rating = y_pred[i][0].item()
            true_rating = ratings[i].item()
            
            # We print the results for a quick sanity check.
            print(f"User: {user_id}, Movie: {movie_id}, Pred: {pred_rating}, True: {true_rating}")
            # We add the (predicted, true) rating tuple to the list for the corresponding user.
            user_movie_test[user_id].append((pred_rating, true_rating))
#%% Precision and Recall
# Now we'll calculate the actual precision and recall metrics.

# We create dictionaries to store the precision and recall for each user.
precisions = {}
recalls = {}

# We define `k`, which is the "at K" in Precision@k. It means we only consider the top K recommendations for each user.
k = 10
# We define our relevance threshold. Any movie with a true rating >= 3.5 is considered "relevant".
thres = 3.5

# We iterate through each user and their list of ratings we just collected.
for uid, user_ratings in user_movie_test.items():
    # For each user, we sort their rated movies in descending order based on our model's PREDICTED rating.
    # This simulates creating a ranked list of recommendations for the user.
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    # We count the total number of truly relevant items for this user in the entire test set.
    n_rel = sum((rating_true >= thres) for (_, rating_true) in user_ratings)

    # We count how many of the top K recommended items we would actually show to the user (i.e., those with a predicted rating above our threshold).
    n_rec_k = sum((rating_pred >= thres) for (rating_pred, _) in user_ratings[:k])

    # This is the key part: we count how many items in the top K recommendations are BOTH recommended (predicted >= thres) AND truly relevant (true_rating >= thres).
    n_rel_and_rec_k = sum(
        ((rating_true >= thres) and (rating_pred >= thres))
        for (rating_pred, rating_true) in user_ratings[:k]
    )

    # A print statement for debugging and inspection.
    print(f"uid {uid},  n_rel {n_rel}, n_rec_k {n_rec_k}, n_rel_and_rec_k {n_rel_and_rec_k}")

    # Precision@k = (Number of recommended items in top K that are relevant) / (Total number of recommended items in top K)
    # This measures: "Of the items we recommended, how many were actually good?"
    precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

    # Recall@k = (Number of recommended items in top K that are relevant) / (Total number of relevant items)
    # This measures: "Of all the good items, how many did we successfully recommend?"
    recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

# Finally, we average the precision and recall scores across all users to get a single performance metric for the whole model.
print(f"Precision @ {k}: {sum(precisions.values()) / len(precisions)}")

print(f"Recall @ {k} : {sum(recalls.values()) / len(recalls)}")
# %%
