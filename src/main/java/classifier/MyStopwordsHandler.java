package classifier;

import java.io.Serializable;
import java.util.List;
import utility.IOManager;
import weka.core.stopwords.StopwordsHandler;

public class MyStopwordsHandler  implements StopwordsHandler, Serializable {

    private List<String> myStopwords;

    public MyStopwordsHandler(String stopWordFile) {
        myStopwords =  IOManager.readFromFile(stopWordFile);
    }

    @Override
    public boolean isStopword(String word) {
        return myStopwords.contains(word); 
    }

}

