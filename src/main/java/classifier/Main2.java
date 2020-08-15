/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import static utility.IOManager.writeToCsvFile;
import java.io.IOException;
import static java.lang.System.exit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.SpreadSubsample;


import static utility.IOManager.*;
import java.io.*;
import static java.lang.System.exit;
import java.util.*;
import utility.IOManager;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.stopwords.StopwordsHandler;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class Main2 {
    
    private static final String tweetsCSV = "file/tweetsVirgola.csv";// "file/testsnellito.csv";
    private static final String stopWordFile = "file/stopWord";
    private static final String stemmerCSV = "file/stemmer.csv";
    
    private static int numberTweets;
    private static String[] tweets;
    private static String[] classes;
    
    private static HashMap<String, Double> attributeWeights; // Peso per ogni features selezionata
    private static List<Attribute> attributes; // Features selezionate
    private static AttributeSelection attSelect;
    private static Double classificatorAccuracy;
    
    

    
    public static void main(String[] args) throws IOException, Exception {
        
        System.out.println("Main 2");
        //Import Tweets
        List<String[]> classList = IOManager.readFromCsvFile(",",tweetsCSV);
 
        numberTweets = classList.size();
        tweets = new String[numberTweets];
        classes = new String[numberTweets];
        FeaturesExtractor fe = new FeaturesExtractor();
        
        int tweetNumber = 0;
        //Concatenate the 5 bit of class
        for(String[] s : classList){
            classes[tweetNumber] = s[1];
            tweets[tweetNumber] = s[0]; 
            tweetNumber++;
        } 

        //Preprocessing
        tweets = fe.deleteUrlAndPic(tweets);    
                
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

        Instances data = source.getDataSet();
        
        // Converto il tipo dei tweets da nominal a string cosi da poter appplicare StringToWordVector
        String[] nsOptions = {"-C", "1"};
        NominalToString nsFilter = new NominalToString();
        nsFilter.setInputFormat(data);
        nsFilter.setOptions(nsOptions);
        data = Filter.useFilter(data, nsFilter);
        
        // Selezione Class come attributo classe
        data.setClassIndex(data.numAttributes() - 1);  
        
        
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
        filter.setInputFormat(data);
        filter.setWordsToKeep(100000); //set to max value to include the max number of features
        filter.setLowerCaseTokens(true);
        filter.setDoNotOperateOnPerClassBasis(true);
        filter.setIDFTransform(true); //Inverse document frequency
        filter.setTFTransform(true);  // term frequency
        
        /*WordsFromFile stopwords = new WordsFromFile();
        stopwords.setStopwords(new File("file/stopWord"));*/
        
        filter.setStopwordsHandler(stopwords); //external stop words file

        //for stemming the data	
        			
        String[] stemOptions = {"-S", "italian"};
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setOptions(stemOptions);
        filter.setStemmer(stemmer);
        
        data = Filter.useFilter(data, filter);
        
        // Dividere training e test
        // stemming filtering
        
        /*
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("debug.arff"));
        saver.writeBatch();
        */
        
        
        
        
        
         
        int seed = 1;          // the seed for randomizing the data
        //int folds = 10;         // the number of folds to generate, >=2
        Double acc = 88.0;
        String accMatrix = "";
        Double sens = 0.0;
        String sensMatrix = "";
        String accTxt = "";
        String sensTxt = "";

        
        for(int folds = 2; folds<=10;folds++){
            System.out.println("Fold: " + folds);   
            Random rand = new Random(seed);   // create seeded number generator
            Instances randData = new Instances(data); // create copy of original data
            randData.randomize(rand);         // randomize data with number generator

            randData.stratify(folds);
            LibSVM svm;
        
            //System.out.println("---    n: " + n);
            
                //System.out.println("---        nu: " + nu);
            for(int n = 50; n<4500; n+=50){ 
                if(n%500 == 0){
                   System.out.println("     Fold: " + folds + " n: " + n);
                }
                for(Double nu = 0.05; nu<0.93;nu+=0.05 ){                  
                    Evaluation eval = new Evaluation(randData);
                    for (int nFold = 0; nFold < folds; nFold++) {
                        Instances train = randData.trainCV(folds, nFold, rand);
                        Instances test = randData.testCV(folds, nFold);

                        train = balancingDataset(train);


                        InfoGainAttributeEval evalRank = new InfoGainAttributeEval();
                        Ranker search = new Ranker();
                        //search.setOptions(new String[] { "-T", "0.0001" });	// information gain threshold
                        search.setNumToSelect(n-1); // Seleziono i primi n elementi
                        attSelect = new AttributeSelection();
                        attSelect.setEvaluator(evalRank);
                        attSelect.setSearch(search);

                        // apply attribute selection
                        attSelect.SelectAttributes(train);

                        // remove the attributes not selected in the last run
                        train = attSelect.reduceDimensionality(train);
                        test = attSelect.reduceDimensionality(test);            



                        svm = new LibSVM();
                        // Sono opzioni prese da internet, verificare
                        String options = ( "-S 1 -K 0 -D 1 -G 0.1 -R 0.0 -N "+nu+" -M 40.0 -C 1 -E 0.001 -P 0.1 -W 1" );
                        String[] optionsArray = options.split( " " );
                        svm.setOptions( optionsArray );
                        svm.buildClassifier(train);

                        eval.evaluateModel(svm, test);

                        /*System.out.println("NU: " + nu );
                        System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
                        System.out.println(eval.toMatrixString());
                        System.out.println("Sensitivity(TP/P) class " + eval.truePositiveRate(1));
                        System.out.println("Specificity(TN/N) class " + eval.trueNegativeRate(1));*/
                    }      
                    if(eval.pctCorrect()>90.0 && sens<eval.truePositiveRate(1)){                        
                        acc = eval.pctCorrect();
                        sens = eval.truePositiveRate(1);
                        accMatrix = eval.toMatrixString();
                        accTxt =" | folds: " + folds +
                                " | nu: " +nu +
                                " | n: " + n +
                                " | Sensitivity: " + eval.truePositiveRate(1) +
                                " | Specificity: " + eval.trueNegativeRate(1) +
                                " | Accuracy: " + acc; 
                        System.out.println();
                        System.out.println();
                        System.out.println(accMatrix);
                        System.out.println(accTxt);
                        System.out.println();
                        System.out.println();
                    }           
                    
                }
            }
                
            
                
    }
        System.out.println();
        System.out.println();
        
        System.out.println("------- Max Accuracy-------");
        System.out.println(accTxt);
        System.out.println(accMatrix);
        
        System.out.println();
        System.out.println();
        /*
        System.out.println("-------Max Sensitivity-------");
        System.out.println(sensTxt);
        System.out.println(sensMatrix);
        */
        System.out.println();
        System.out.println();
        
        
        
    }
    
    private static Instances balancingDataset(Instances data) throws Exception {
        //Balancing dataset
        SpreadSubsample ff = new SpreadSubsample();

        String opt = "-M 1.0 -X 0.0 -S 1";//any options you like, see documentation
        String[] optArray = weka.core.Utils.splitOptions(opt);//right format for the options

        ff.setOptions(optArray);
        ff.setInputFormat(data);


        return Filter.useFilter(data, ff);
    }
    
    
}