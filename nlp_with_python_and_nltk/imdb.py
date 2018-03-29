# imdb-pie is required
# Author: Sunjay Dhama
from imdbpie import Imdb


def help_print_review(review_list):
    for review in review_list:
        try:
            print('ID: ' + review['id'])
            print('Author: ' + review['author']['displayName'])
            print('Author Rating: ' + str(review['authorRating']))
            print('Helpfulness Score: ' + str(review['helpfulnessScore']))
            print('Review Title: ' + review['reviewTitle'])
            print('Spoiler: ' + str(review['spoiler']))
            print('Submission Date: ' + str(review['submissionDate']))
            print('Review Text: ' + review['reviewText'])
            print('**********************************')
            print(' ')
        except:
            print('An error occured')

def help_print_movie(movie_dict):

    for movie in movie_dict:
        try:
            print('Title: ' + movie['title'])
            print('Year: ' + movie['year'])
            print('IMDB ID: ' +movie['imdb_id'])
            print('**********************************')
            print(' ')

        except TypeError:
            print('TypeError')


def main():
    imdb = Imdb()
    movie = str(input('Movie Name: '))
    movie_search = '+'.join(movie.split())
    # print(imdb.search_for_name("Christian Bale"))
    movie_dict  = imdb.search_for_title(movie_search)
    help_print_movie(movie_dict)
    imdb_id = str(input('IMBD ID: '))
    review_dict = imdb.get_title_user_reviews(imdb_id)
    review_list = review_dict['reviews']
    help_print_review(review_list)

if __name__ == '__main__':
    main()
