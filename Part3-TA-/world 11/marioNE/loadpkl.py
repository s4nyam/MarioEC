import pickle

# Specify the path to your .pkl file and the path for the text file
pkl_file_path = 'neat-checkpoint-13.pkl'
txt_file_path = 'winner.txt'

try:
    with open(pkl_file_path, 'rb') as pkl_file, open(txt_file_path, 'w') as txt_file:
        data = pickle.load(pkl_file)
        # Now 'data' contains the data loaded from the pickle file
        txt_file.write(str(data))
    
    print("Data loaded and saved to a text file successfully.")
except FileNotFoundError:
    print(f"File '{pkl_file_path}' not found.")
except Exception as e:
    print(f"An error occurred while loading the data from '{pkl_file_path}': {str(e)}")
