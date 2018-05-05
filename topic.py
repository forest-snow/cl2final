import pickle
from preprocessing import User, Post
from nltk.corpus import stopwords


stopWords = set(stopwords.words('english'))

def read_ids(file_):
    with open(file_, 'r') as f:
        data = f.read().splitlines()
        ids = [int(i) for i in data]
    return ids

def get_user_ids(path1, path2, file_):
    ids1 = read_ids(path1+file_)
    ids2 = read_ids(path2+file_)
    return ids1 + ids2

def get_users(user_dict, ids):
    new_dict = dict((i, user_dict[i]) for i in ids if i in user_dict)
    return new_dict

if __name__ == '__main__':
    file_ = "/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/user_info"
    path_control = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/'
    path_sw = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/'
    output_path = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/'


    pickle_obj = open(file_, 'rb')
    user_dict = pickle.load(pickle_obj)
    docs = []
    n_posts = 0
    for n_users, (_,user) in enumerate(user_dict.items()):
        if n_users % 1000 == 0:
            print('User {}'.format(n_users))

        for post in user.posts:
            if n_posts % 1000 == 0:
                print('Post {}'.format(n_posts))
            text = post.text
            tokens = text.split()
            new_tokens = [token for token in tokens if token not in stopWords]
            new_text = ' '.join(new_tokens)
            post.text = new_text
            n_posts += 1

    print('User {}'.format(n_users))
    print('Post {}'.format(n_posts))

    train_ids = get_user_ids(path_control, path_sw, 'TRAIN.txt') 
    test_ids = get_user_ids(path_control, path_sw, 'TEST.txt') 
    dev_ids = get_user_ids(path_control, path_sw, 'DEV.txt') 

    train_users = get_users(user_dict, train_ids)
    test_users = get_users(user_dict, test_ids)
    dev_users = get_users(user_dict, dev_ids)

    with open(output_path+'train_users', 'wb') as f:
        pickle.dump(train_users, f)

    with open(output_path+'dev_users', 'wb') as f:
        pickle.dump(dev_users, f)

    with open(output_path+'test_users', 'wb') as f:
        pickle.dump(test_users, f)
