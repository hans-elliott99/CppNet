

#include "data/dataset.h"
#include "utils/functions.h"
#include "utils/matrix.h"


int main()
{
    Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");
    data.head(5);
    data.make_split(0.8);

    std::vector<size_t> xtrshape {data.shape(DataSplit::TRAIN, false, true)};
    
    Matrix Xtr(xtrshape[0], xtrshape[1]);

    // Xtr = data.toMatrix(); //Error here...

    // Xtr.shape();  
    // std::cout << Xtr.M.size(); 
   
}