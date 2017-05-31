## [Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)

<p align="justify">
This paper proposes a model that can align books to their movie releases to provide rich descriptive explanations for visual content. To align movies and books, they used a nueral sentence embedding model that is trained in an unsupervised way from a large corpus of books. To compute similarities between movie clips and sentences in the book, they used video-text neural embedding. 
<p align="justify">

#### Key Points

- To align a movie with a book by exploiting visual information as well as dialogs, they proposed a pairwise Conditional Random Field (CRF) that smooths the alignments by encouraging them to follow a linear timeline.
- To compute similarity between two sentences, they used [skip-thought vectors](http://papers.nips.cc/paper/5950-skip-thought-vectors.pdf).
- To compute similarities between movie shots and sentences, they followed the image-sentence ranking model proposed by [Kiros et al. 2014](http://www.cs.toronto.edu/~rkiros/papers/mnlm2014.pdf).
- To extract frame features, they used [GoogLeNet](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) architecture and [hybrid-CNN](http://papers.nips.cc/paper/5349-learning-deep-features-for-scene-recognition-using-places-database.pdf).
- They designed a context aware similarity measure that takes into account both sentence to sentence similarity and shots to sentence similarity with a fixed context window in both to predict a new similarity score.
- They formulated the movie/book alignment problem as inference in a Conditional Random Field that encourages nearby shots/dialog alignments to be consistent.

#### Notes

<p align="justify">
They used movie description dataset from <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Rohrbach_A_Dataset_for_2015_CVPR_paper.pdf">A Dataset for Movie Description</a> to learn visual-semantic embedding. To evaluate the book-movie alignment model, they collected a dataset with 11 movie/book pairs annotated with 2,070 shot-to-sentence correspondence. In terms of training speed, their video-text model watches 1,440 movies per day and sentence model reads 870 books per day.
<p align="justify">

**Expermental Dataset**: [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb), [MovieBook Dataset](http://yknzhu.wixsite.com/mbweb)

**More details on this work**: [Project Page](http://yknzhu.wixsite.com/mbweb), [Extended Version of the Paper](http://www.cs.toronto.edu/~zemel/documents/align.pdf)

**Bibliography**
```
@inproceedings{zhu2015aligning,
  title={Aligning books and movies: Towards story-like visual explanations by watching movies and reading books},
  author={Zhu, Yukun and Kiros, Ryan and Zemel, Rich and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={19--27},
  year={2015}
}
```
