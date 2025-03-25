const coll_movies = "movies" ;
const db = "cinema" ;

const projection = {title : 1,
    actors : 1, 
    year: 1,
    _id: 0
    }

const movie_2000_210 = db.movies.find(
    ///filtre
{
    year: 
    {$gt:2000,
        $lt:2010
    }

},

projection
// Ce que l'on veut collecter en output
)

console.log(movie_2000_210)
