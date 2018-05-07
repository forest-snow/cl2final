import pickle
from preprocessing import User, Post
import gensim

if __name__ == '__main__':
    path = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/'
    pickle_obj = open(path+'test_users', 'rb')
    user_dict = pickle.load(pickle_obj)
    print(user_dict)
    docs = []

    n_posts = 0
    for n_users, (_,user) in enumerate(user_dict.items()):
        if n_users % 1000 == 0:
            print('User {}'.format(n_users))

        for post in user.posts:
            if n_posts % 1000 == 0:
                print('Post {}'.format(n_posts))
            text = post.text
            docs.append(text)
            n_posts += 1

    print('User {}'.format(n_users))
    print('Post {}'.format(n_posts))
