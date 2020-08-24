package classifier;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import utility.ClassifierData;
import weka.core.converters.ConverterUtils;
import utility.IOManager;
import static utility.IOManager.writeToCsvFile;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import org.tartarus.snowball.SnowballProgram;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.StopwordsHandler;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class Classificator10Crossfold {
    /* START Parameter */
    private static final String rowTweetsCSV = "file/tweets_ripuliti.csv";//"file/tweetsVirgola.csv"; //formato file: tweet,classe
    private static final String stopWordFile = "file/stopWord";    
    
    private static int selectedClassifier = 0;  /*  (0) DecisionTree 
                                                    (1) SVM 
                                                    (2) MultinomialNB 
                                                    (3) kNN 
                                                    (4) Adaboost 
                                                    (5) RandomForest
                                                */
    /* END Parameter */
    
    /* START Da esportare assieme al classificatore per il preprocessing */
    private static final String[] keyWord = 
    {
        "fogne","fognatura","fognaria","allagato","allagata",
        "allagamento","alluvione","caditoia","caditoie","tubatura",
        "tubature","sommerse","tombino","tombini","scolo",
        "allerta meteo","maltempo"
    };  
    /* END Da esportare assieme al classificatore per il preprocessing */
    
    private static Instances trainSet, testSet, data;
    private static int numberTweets;
    private static String[] tweets;
    private static String[] classes;
    private static double[] accuracyFold;
    private static double[] fscoreFold;
    private static double[] AUCFold;
    private static String[] matrixFold;
    private static Classifier[] classifierFold;
    private static String classifierName = "";

    
    public static void main(String[] args) throws IOException, Exception {
        System.out.println("Classifier [START]");
        
        // Importo i tweet             
        importTweets(rowTweetsCSV);
        
        accuracyFold = new double[10];
        fscoreFold = new double[10];
        AUCFold = new double[10];
        matrixFold = new String[10];
        classifierFold = new Classifier[10];
        
        // Ripeto per le 10 fold dalvandomi i risultati ad ogni istanza
        for(int fold=1; fold<=10; fold++){
            // Ottengo test set e training set
            splitDataset(fold); 

            // Bilancio il train set
            trainSet = classInstancesBalancing(trainSet);        

            // Uso un classificatore
            textClassification(fold-1);
        }  
        
        System.out.println("\nClassificator: " + classifierName);
        printResultIndex();
        chooseBestClassifier();

        System.out.println("Classifier [END]");
    }
    
    
    private static void importTweets(String path) throws Exception {
        // Il file path è stato preparato manualmente mediante excel
        List<String[]> classList = IOManager.readFromCsvFile(",",path);
 
        numberTweets = classList.size();
        tweets = new String[numberTweets];
        classes = new String[numberTweets];
        
        int tweetNumber = 0;
        //Concatenate the 5 bit of class
        for(String[] s : classList){
            classes[tweetNumber] = s[1];
            tweets[tweetNumber] = s[0]; 
            tweetNumber++;
        } 
        
        tweets = deleteUrlAndPic(tweets);    
                
        List<String[]> out = new ArrayList<>();
        String[] tmp = new String[2];
        
        tmp[0] = "tweet";
        tmp[1] = "classe";
        out.add(tmp);
        
        for(int pos = 0; pos<numberTweets; pos++){
            tmp = new String[2];
            tmp[0] = "'"+tweets[pos]+"'";
            tmp[1] = classes[pos];            
            out.add(tmp);
        }
      
        writeToCsvFile(out,",","tmp.csv");
        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("tmp.csv");

        data = source.getDataSet();
        
        // Converto il tipo dei tweets da nominal a string cosi da poter applicare StringToWordVector
        String[] nsOptions = {"-C", "1"};
        NominalToString nsFilter = new NominalToString();
        nsFilter.setInputFormat(data);
        nsFilter.setOptions(nsOptions);
        data = Filter.useFilter(data, nsFilter);
        
        // Selezione Class come attributo classe
        data.setClassIndex(data.numAttributes() - 1); 
    }
    
    private static void splitDataset(int fold) throws Exception {
        /* Faccio stratified in modo da mantenera la distribuzione, divido in 5 folds
           cosi 3 fanno parte del train (75%) e 1 fa parte del test (25%)
        */
        // Estraggo train set
        StratifiedRemoveFolds strRmvFoldsTrain = new StratifiedRemoveFolds();
        String optionsSRFTrain = ( "-S 0 -N 10 -F "+fold+" -V");
        String[] optionsArrayTrain = optionsSRFTrain.split( " " );
        strRmvFoldsTrain.setOptions(optionsArrayTrain);
        strRmvFoldsTrain.setInputFormat(data);
        trainSet = StratifiedRemoveFolds.useFilter(data, strRmvFoldsTrain);

        // Estraggo test set
        StratifiedRemoveFolds strRmvFoldsTest = new StratifiedRemoveFolds();
        String optionsSRFTest = ( "-S 0 -N 10 -F "+fold);
        String[] optionsArrayTest = optionsSRFTest.split( " " );
        strRmvFoldsTest.setOptions(optionsArrayTest);
        strRmvFoldsTest.setInputFormat(data);
        testSet = StratifiedRemoveFolds.useFilter(data, strRmvFoldsTest);
    }

    private static StringToWordVector getTextElaborationFilter() throws Exception {
        // Creo l'handler per le StopWord
        List<String> stopWords =  IOManager.readFromFile(stopWordFile);        
        StopwordsHandler stopwords = new StopwordsHandler() {

            @Override
            public boolean isStopword(final String word) {
                return stopWords.contains(word);
            }
        };
        
        //  Applico tokenization, stop word e stemming
        StringToWordVector filter = new StringToWordVector();
        filter.setAttributeIndices("1");
        filter.setTokenizer(new WordTokenizer());
        filter.setWordsToKeep(100000); //set to max value to include the max number of features
        filter.setLowerCaseTokens(true);
        filter.setDoNotOperateOnPerClassBasis(true);
        filter.setIDFTransform(true); //Inverse document frequency
        filter.setTFTransform(true);  // term frequency
        
        /*WordsFromFile stopwords = new WordsFromFile();
        stopwords.setStopwords(new File("file/stopWord"));*/
        
        filter.setStopwordsHandler(stopwords); //external stop words file

        //for stemming the data          
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("italian");
        filter.setStemmer(stemmer);
        return filter;
    }
    
    private static  AttributeSelection getAttributeSelectionFilter() throws Exception {
        Ranker ranker = new Ranker();
        ranker.setThreshold(0);	// information gain threshold
        AttributeSelection as= new AttributeSelection();
        as.setEvaluator(new InfoGainAttributeEval());
        as.setSearch(ranker);
        return as;
    }

    private static void textClassification(int fold) throws Exception {
        FilteredClassifier classifier = new FilteredClassifier();
        //MultiFilter mf = new MultiFilter();
        //mf.setFilters(new Filter[] { getTextElaborationFilter(), getAttributeSelectionFilter()});
        
        MultiFilter mf = new MultiFilter();
        Filter[] filters = new Filter[2];
        filters[0] = getTextElaborationFilter();
        filters[1] = (Filter) getAttributeSelectionFilter();
        mf.setFilters(filters);

        classifier.setFilter(mf);
        
        Classifier classifierSelected = null;

        switch(selectedClassifier){
            case 0:                
                classifierSelected = doDecisionTree();
                classifierName = "DecisionTree(10CV)";
                break;
                
            case 1:                
                classifierSelected = doSVM();
                classifierName = "SVM(10CV)";
                break;
                
            case 2:                
                classifierSelected = doMultinomialNB();
                classifierName = "MultinomialNB(10CV)";
                break;
                
            case 3:                
                classifierSelected = dokNN();
                classifierName = "kNN(10CV)";
                break;
                
            case 4:                
                classifierSelected = doAdaboost();
                classifierName = "Adaboost(10CV)";
                break;
                
            case 5:                
                classifierSelected = doRandomForest();
                classifierName = "RandomForest(10CV)";
                break;
                
            default:
                // Non è stato selezionato un classificatore o ne è stato sbagliato il nome
                System.out.println("[ERROR] Wrong classificator selected, check paramiter");
                System.exit(1);            
        } 
         
        classifier.setClassifier(classifierSelected);
        classifier.buildClassifier(trainSet);
        
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(classifier, testSet);
        
        saveFoldResult(fold,eval,classifier);
        
        // TODO non posso esportarlo perchè Snowball Stemmer non è serializzabile
        //exportClassifier(classifier, classifierName);
    }
    
    private static void printIndex(Evaluation eval) throws Exception{
        double acc = eval.pctCorrect();
        double precision = eval.truePositiveRate(1);                        
        double fscore = eval.fMeasure(1);
        if(Double.isNaN(fscore)){
            // Con Recall e precision == 0.0 da nan quindi va settato
            fscore = 0.0;
        }
        double recall = eval.recall(1);
        double auc = eval.areaUnderROC(1);
        String confusionMatrix = eval.toMatrixString();
        
        String index =  "\t| Accuracy: " + acc +
                        "\n" +
                        "\t| Precision: " + precision*100 +
                        "\n" +
                        "\t| F-Score: " + fscore*100 +
                        "\n" +
                        "\t| Recall: " + recall*100 +
                        "\n" +
                        "\t| AUC: " + auc*100 +
                        "\n";
        
        System.out.println(index);
        System.out.println(confusionMatrix); 
    }
  
    private static Classifier doDecisionTree() throws Exception {        
        String [] options =new String[1];
        options[0] = "-U";
        J48 classificator = new J48();
        classificator.setOptions(options);
        return (Classifier) classificator;
    }
    
    private static Classifier doSVM() throws Exception {
        LibSVM classificator= new LibSVM();
        String options = ( "-S 0 -K 0 -D 1 -G 0.1 -R 0.0 -N 0.2 -M 40.0 -C 1 -E 0.001 -P 0.1 -W 1" );
        String[] optionsArray = options.split( " " );
        classificator.setOptions( optionsArray );
        return (Classifier) classificator;
    }
    
    private static Classifier doMultinomialNB() throws Exception {
        NaiveBayesMultinomial classificator = new NaiveBayesMultinomial();
        return (Classifier) classificator;
    }
    
    private static Classifier dokNN() throws Exception {        
        IBk classificator = new IBk();
        String options = "-K 1";
        String[] optionsArray = options.split( " " );
        classificator.setOptions( optionsArray );
        return (Classifier) classificator;
    }
    
    private static Classifier doAdaboost() throws Exception {
        AdaBoostM1 classificator = new AdaBoostM1();
        return (Classifier) classificator;
    }
    
    private static Classifier doRandomForest() throws Exception {
        RandomForest classificator = new RandomForest();
        return (Classifier) classificator;
    }
      
    private static Instances classInstancesBalancing(Instances inst) throws Exception {
        //Balancing dataset
        SpreadSubsample ff = new SpreadSubsample();
        String opt = "-M 1.0 -X 0.0 -S 1";
        String[] optArray = weka.core.Utils.splitOptions(opt);
        ff.setOptions(optArray);
        ff.setInputFormat(inst);        
        return Filter.useFilter(inst, ff);
    }
        
    private static String[] deleteUrlAndPic(String[] tweets){
        String urlRegex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+";
        String picRegex = "pic.twitter.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+";
        for(int i = 0; i<tweets.length; i++){
            tweets[i] = tweets[i].replaceAll(urlRegex, "").replaceAll(picRegex, "").replaceAll("[^a-zA-Z ]", " ").toLowerCase();
        }  
        return tweets;
    }
    
    private static void exportClassifier(FilteredClassifier c, String name) throws IOException, Exception{
        ClassifierData cd = new ClassifierData();
        cd.classificatorName = name;
        cd.classifier = c;
        // Keyword utilizate delezionate dai parametri ad inizio classe
        cd.researchKey = keyWord; 
        //weka.core.SerializationHelper.write("classifier/"+name+".classifier", c); // Nemmeno con questo lo esporta
        IOManager.saveClassBinary("classifier/"+name+".classifier", cd);
    }

    private static void saveFoldResult(int fold, Evaluation eval, FilteredClassifier classifier) throws Exception {
        double fscore = eval.fMeasure(1);
        if(Double.isNaN(fscore)){
            // Con Recall e precision == 0.0 da nan quindi va settato
            fscore = 0.0;
        }
        accuracyFold[fold] = eval.pctCorrect();
        fscoreFold[fold] = fscore*100;
        AUCFold[fold] =  eval.areaUnderROC(1)*100;
        matrixFold[fold] =  eval.toMatrixString();
        classifierFold[fold] =classifier;
    }
    
    private static String printStringVector(double[] values) {
        DecimalFormat df = new DecimalFormat("#.##");
        String ret = "[  ";
        for(double val : values){
            ret += df.format(val) + "  "; 
        }
        return ret + "]";
    }
    
    private static double average(double[] n) {
        double sum = 0;
        for(double value:n){
            sum+=value;
        }
        return sum/n.length;
    }
    
    private static String confidanceInterval(double[] values) {
        double average = average(values);
        double varianceSum = 0.0;
        for (int i = 0; i < values.length; i++) {
            varianceSum += (values[i] - average) * (values[i] - average);
        }
        double variance = varianceSum / (values.length - 1);
        double standardDaviation= Math.sqrt(variance);
        DecimalFormat df = new DecimalFormat("#.##");
        return df.format(average) + " +- " + df.format(1.96 * standardDaviation);
    }

    private static void printResultIndex() {
        System.out.println("Accuracy: \n\t" + confidanceInterval(accuracyFold)+ " \t "  + printStringVector(accuracyFold));
        System.out.println("F-Score: \n\t" + confidanceInterval(fscoreFold)+ " \t"  + printStringVector(fscoreFold));
        System.out.println("AUC: \n\t" + confidanceInterval(AUCFold)+ " \t"  + printStringVector(AUCFold));
        System.out.println("\n");
    }

    private static void chooseBestClassifier() {
        System.out.println("TODO scelta miglior classificatore ed esportazione");
    }

}