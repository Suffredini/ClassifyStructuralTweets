package classifier;


import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import weka.core.converters.ConverterUtils;
import java.util.*;
import utility.ExportedClassifierData;
import utility.IOManager;
import static utility.IOManager.writeToCsvFile;
import weka.attributeSelection.AttributeSelection;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.StopwordsHandler;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class BuildClassificator10Fold {
    /* Parameter */
    private static final String tweetsCSV = "file/tweets_ripuliti.csv";//"file/tweetsVirgola.csv"; //formato file: tweet,classe
    private static final String stopWordFile = "file/stopWord";
    private static final String reportClassificationAccuracy= "analysis/reportAccuracy.csv";
    private static final String reportClassificationPrecision = "analysis/reportPrecision.csv";
    private static final String reportClassificationFscore = "analysis/reportFscore.csv";
    private static final String reportClassificationRecall = "analysis/reportRecall.csv";
    private static final String reportClassificationAUC = "analysis/reportAUC.csv";

    
    private static final String[] classifiers = {"nu-svm", "j48"};
    private static final int selectedClassifier = 0;
    
    private static final String[] keyWord = 
    {
        "fogne","fognatura","fognaria","allagato","allagata",
        "allagamento","alluvione","caditoia","caditoie","tubatura",
        "tubature","sommerse","tombino","tombini","scolo",
        "allerta meteo","maltempo"
    };
    
    private static final int folds = 10; //Included
    
    private static final int featuresStart = 10; //Included
    private static final int featuresEnd = 4000; //Included
    private static final int featuresIncrement = 10;
    
    
    // Only for nu-svm needed
    private static final double nuStart = 0.09; //Included
    private static final double nuEnd = 0.99; //Included
    private static final double nuIncrement = 0.1;
    
    /* End parameter */
    
    
    private static int numberTweets;
    private static String[] tweets;
    private static String[] classes;
    private static Instances data, train, test, trainTmp, testTmp;

    private static AttributeSelection attSelect;
    private static HashMap<String,Double> attributesTaExport;
    

    
    public static void main(String[] args) throws IOException, Exception {
        System.out.println(classifiers[selectedClassifier]+"_Classifier -> Start");
        
        importTweets();
        preprocessingTweets();
        
        // Miglior combinazioine n e nu
        int nSelected = 0;
        double nuSelected = 0.0;
        double avgAccGlobal = 0.0;
        double[] accGlobal = new double[folds];
        double[] precisionGlobal = new double[folds];
        double[] fscoreGlobal = new double[folds];
        double[] recallGlobal = new double[folds];
        double[] aucGlobal = new double[folds];
        String[] cMatrixGlobal = new String[folds];
        Classifier[] classifierGlobal = new Classifier[folds];
        
        
        // Dati fissando n e nu singola istanza, per verificare se trovo una coimbinazione migliore
        double avgAcc = 0.0;
        double[] acc = new double[folds];
        double[] precision = new double[folds];                     
        double[] fscore = new double[folds];
        double[] recall = new double[folds];
        double[] auc = new double[folds];
        String[] cMatrix = new String[folds];
        Classifier[] classifier = new Classifier[folds];
        
        List<String[]> outAccuracy = initializeList();                    
        List<String[]> outPrecision = initializeList(); 
        List<String[]> outFscore = initializeList(); 
        List<String[]> outRecall = initializeList(); 
        List<String[]> outAUC = initializeList(); 
        

      
        ExportedClassifierData ecd = new ExportedClassifierData();
        ecd.researchKey = keyWord;
        
        System.out.println("-- build classifier -> start");
        System.out.println("\n");

        if(classifiers[selectedClassifier].equals("nu-svm")){
            for(int n = featuresStart; n<=featuresEnd; n+=featuresIncrement){ 
                System.out.println("Inizio n: " + n);
                for(double nu = nuStart; nu<=nuEnd;nu+=nuIncrement ){                    
                    for(int fold = 0; fold<folds; fold++){                           
                        extractionTrainAndTestSet(folds, fold+1);
                        // Balancing train set
                        train = balancingDataset(train);
                        featuresSelection(n);
                        // Classifico
                        LibSVM svm = new LibSVM();
                        String options = ( "-S 1 -K 0 -D 1 -G 0.1 -R 0.0 -N "+nu+" -M 40.0 -C 1 -E 0.001 -P 0.1 -W 1" );
                        String[] optionsArray = options.split( " " );
                        svm.setOptions( optionsArray );
                        svm.buildClassifier(trainTmp);

                        Evaluation eval = new Evaluation(trainTmp);
                        eval.evaluateModel(svm, testTmp);

                        acc[fold] = eval.pctCorrect();
                        precision[fold] = eval.truePositiveRate(1);                        
                        fscore[fold] = eval.fMeasure(1);
                        recall[fold] = eval.recall(1);
                        auc[fold] = eval.areaUnderROC(1);
                        cMatrix[fold] = eval.toMatrixString();
                        classifier[fold] = svm;

                        
                        
                    }
                    //Vedo se la combinazione ne nu da un miglior risultato
                    /*
                        // DEBUG
                        System.out.println("Accuracy: "+printStringVector(acc) + " AVG: " + average(acc));
                        System.out.println("Fscore: "+printStringVector(fscore) + " AVG: " + average(fscore));
                        System.out.println("FscoreGlobal: "+printStringVector(fscoreGlobal) + " AVG: " + average(fscoreGlobal));
                    */
                    
                    avgAcc = average(acc);
                    
                    double avgPrec = average(precision);
                    double avgFsc = average(fscore);
                    double avgRec = average(recall);
                    double avgAuc = average(auc);
                    
                 
                    outAccuracy.add(prepareListToexportDataToCsv(folds,n,nu,acc,avgAcc));                    
                    outPrecision.add(prepareListToexportDataToCsv(folds,n,nu,precision,avgPrec));
                    outFscore.add(prepareListToexportDataToCsv(folds,n,nu,fscore,avgFsc));
                    outRecall.add(prepareListToexportDataToCsv(folds,n,nu,recall,avgRec));
                    outAUC.add(prepareListToexportDataToCsv(folds,n,nu,auc,avgAuc));
      
                    
                    //if( (average(acc)>80) && (average(fscore) > average(fscoreGlobal))){
                    //System.out.println(" - AVG Accuracy: " + avgAcc);
                    if(avgAcc > avgAccGlobal){   
                        System.out.println("n: " + n + " - nu: "+nu);
                        System.out.println(" - AVG Accuracy: " + avgAcc);
                        System.out.println(" - AVG Fscore: " + avgFsc);
                        System.out.println("\n");
                        
                        nSelected = n;
                        nuSelected = nu;
                        avgAccGlobal = avgAcc;
                        accGlobal = acc;
                        precisionGlobal = precision;                     
                        fscoreGlobal = fscore;
                        recallGlobal = recall;
                        aucGlobal = auc;
                        cMatrixGlobal = cMatrix;
                        classifierGlobal = classifier;
                        ecd.attributes = attributesTaExport;
                    }
                    
                    acc = new double[folds];
                    precision = new double[folds];                     
                    fscore = new double[folds];
                    recall = new double[folds];
                    auc = new double[folds];
                    cMatrix = new String[folds];
                    classifier = new Classifier[folds];
                }
            } 
        } else if(classifiers[selectedClassifier].equals("j48")){
            System.out.println("not implemented");
            /*
            String [] options =new String[1];
            options[0] = "-U";
            J48 J48Classificator = new J48();
            J48Classificator.setOptions(options);
            J48Classificator.buildClassifier(trainTmp);

            Evaluation eval = new Evaluation(trainTmp);
            eval.evaluateModel(J48Classificator, testTmp);

            if(eval.pctCorrect()>0.80 && fscore<eval.fMeasure(1)){
                acc = eval.pctCorrect();
                precision = eval.truePositiveRate(1);                        
                fscore = eval.fMeasure(1);
                recall = eval.recall(1);
                auc = eval.areaUnderROC(1);
                accMatrix = eval.toMatrixString();

                accTxt =" | folds: " + folds +
                        "\n" +
                        " | fold: " + fold +
                        "\n" +
                        " | n: " + n +
                        "\n" +
                        " | Accuracy: " + acc +
                        "\n" +
                        " | Precision: " + precision +
                        "\n" +
                        " | F-Score: " + fscore +
                        "\n" +
                        " | Recall: " + recall +
                        "\n" +
                        " | AUC: " + auc;

                ecd.classifier = J48Classificator;
                ecd.attributes = attributesTaExport;
                ecd.resume = accTxt;
                ecd.confusionMatrix = eval.toMatrixString();
            }*/
        } else {
            System.out.println("ERROR: wrong classifier selected, check in parameters");
            System.exit(1);
        }    
        
        writeToCsvFile(outAccuracy,";",reportClassificationAccuracy);
        writeToCsvFile(outPrecision,";",reportClassificationPrecision);
        writeToCsvFile(outFscore,";",reportClassificationFscore);
        writeToCsvFile(outRecall,";",reportClassificationRecall);
        writeToCsvFile(outAUC,";",reportClassificationAUC);
        
        // Valori medi classificatori migliori tra combo ne nu
        String bestCombination =   
                    " | folds: " + folds +                                        
                    "\n" +                                  
                    " | n: " + nSelected +                                        
                    "\n" +                                       
                    " | nu: " + nuSelected +                                        
                    "\n" +
                    " | Accuracy: " + printStringVector(accGlobal) +
                    "\n" +
                    " | Precision: " +  printStringVector(precisionGlobal) +
                    "\n" +
                    " | F-Score: " + printStringVector(fscoreGlobal) +
                    "\n" +
                    " | Recall: " + printStringVector(recallGlobal) +
                    "\n" +
                    " | AUC: " + printStringVector(aucGlobal);  
        
        // Miglior classificatore tra le combo n e nu selezionate
        int bestFold = -1;
        double accuracyTmp = 0.0;
        for(int i=0; i<accGlobal.length; i++){
            if(accuracyTmp < accGlobal[i]){
                accuracyTmp = accGlobal[i];
                bestFold = i;
            }
        }
        
        if(bestFold == -1){
            System.out.println("ERROR: can't select classifier");
            System.exit(1);
        }
        
        String bestSingleClassifier =   
                    " | folds: " + folds + 
                    "\n" +                                       
                    " | n: " + nSelected +                                        
                    "\n" +                                       
                    " | nu: " + nuSelected +
                    "\n" +                                       
                    " | fold: " + (bestFold+1) +   
                    "\n" +
                    " | Accuracy: " + accGlobal[bestFold] +
                    "\n" +
                    " | Precision: " +  precisionGlobal[bestFold] +
                    "\n" +
                    " | F-Score: " + fscoreGlobal[bestFold] +
                    "\n" +
                    " | Recall: " + recallGlobal[bestFold] +
                    "\n" +
                    " | AUC: " + aucGlobal[bestFold];   
        ecd.classifier = classifierGlobal[bestFold];
        ecd.attributes = attributesTaExport;  
        ecd.confusionMatrix = cMatrixGlobal[bestFold];
        ecd.resume = bestSingleClassifier;
        
     
        
        //IOManager.saveClassBinary("classifier/nu-svm.classifier", ecd);
        IOManager.saveClassBinary("classifier/"+ classifiers[selectedClassifier]+".classifier", ecd);
        System.out.println();
        
        System.out.println("------- Best -------");
        System.out.println(bestCombination);
        System.out.println();        
        System.out.println(bestSingleClassifier);
        System.out.println();
        System.out.println(cMatrixGlobal[bestFold]);        
        System.out.println();
 
        System.out.println("-- build classifier -> end");
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

    private static void importTweets() {
        System.out.println("- import tweets -> start");
        //Import Tweets
        List<String[]> classList = IOManager.readFromCsvFile(",",tweetsCSV);
 
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
        System.out.println("-- import tweets -> end");
    }

    private static void preprocessingTweets() throws Exception {
        System.out.println("- preprocessing -> start");
        //Preprocessing
        FeaturesExtractor fe = new FeaturesExtractor();
        
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

        data = source.getDataSet();
        
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
        System.out.println("-- preprocessing -> end");
    }

    private static void extractionTrainAndTestSet(int folds, int fold) throws Exception {
        // Estraggo train set
        StratifiedRemoveFolds strRmvFoldsTrain = new StratifiedRemoveFolds();
        String optionsSRFTrain = ( "-S 0 -N "+folds+" -F "+fold+" -V");
        String[] optionsArrayTrain = optionsSRFTrain.split( " " );
        strRmvFoldsTrain.setOptions(optionsArrayTrain);
        strRmvFoldsTrain.setInputFormat(data);
        train = StratifiedRemoveFolds.useFilter(data, strRmvFoldsTrain);

        // Estraggo test set
        StratifiedRemoveFolds strRmvFoldsTest = new StratifiedRemoveFolds();
        String optionsSRFTest = ( "-S 0 -N "+folds+" -F "+fold);
        String[] optionsArrayTest = optionsSRFTest.split( " " );
        strRmvFoldsTest.setOptions(optionsArrayTest);
        strRmvFoldsTest.setInputFormat(data);
        test = StratifiedRemoveFolds.useFilter(data, strRmvFoldsTest);
    }

    private static void featuresSelection(int n) throws Exception {
        // Features selection
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
        trainTmp = attSelect.reduceDimensionality(train);
        testTmp = attSelect.reduceDimensionality(test); 


        // Estraggo gli attributi per esportarli con il classificatore
        Enumeration<Attribute> tt = trainTmp.enumerateAttributes();                
        Attribute a;
        attributesTaExport = new HashMap<>();
        double weight = 0.0;
        while(tt.hasMoreElements()){
            a = tt.nextElement();                 
            double[] ar = trainTmp.attributeToDoubleArray(a.index());
            for(int i = 0; i <ar.length; i++){
                if(ar[i] != 0.0){
                    weight = ar[i];
                    break;
                }
            }
            attributesTaExport.put(a.name(),weight);
        }
    }

    private static double average(double[] n) {
        double sum = 0;
        for(double value:n){
            sum+=value;
        }
        return sum/n.length;
    }

    private static String printStringVector(double[] values) {
        String ret = "[";
        for(double val : values){
            ret += val + " "; 
        }
        return ret + "]";
    }

    private static String setComma(double[] values) {
        String ret = "";
        for(int i = 0; i<values.length-1; i++){
            ret += Double.toString(values[i]).replace(".", ",")+";"; 
        }
        ret += Double.toString(values[values.length-1]).replace(".", ",");
        return ret;
    }

    private static double[] confidanceInterval(double[] values, double average) {
        double varianceSum = 0.0;
        for (int i = 0; i < values.length; i++) {
            varianceSum += (values[i] - average) * (values[i] - average);
        }
        double variance = varianceSum / (values.length - 1);
        double standardDaviation= Math.sqrt(variance);
        
        double lower = average - 1.96 * standardDaviation;
        double higher = average + 1.96 * standardDaviation;
        double[] ci = new double[2];
        
        ci[0] = lower;
        ci[1] = higher;
        
        return ci;
    }

    private static String[] prepareListToexportDataToCsv(int folds, int n, double nu, double[] acc, double avgAcc) {
        double[] ci = confidanceInterval(acc, avgAcc);                                          
        String[] tmp = new String[7];
        
        tmp[0] = ""+folds;
        tmp[1] = ""+n;  
        tmp[2] = ""+nu;  
        tmp[3] = setComma(acc); // Composto da più valori separati da virgola 
        tmp[4] = Double.toString(avgAcc).replace(".", ",");  
        tmp[5] = Double.toString(ci[0]).replace(".", ",");
        tmp[6] = Double.toString(ci[1]).replace(".", ",");
        return tmp;
    }

    private static List<String[]> initializeList() {
        List<String[]> out = new ArrayList<>(); 
        String[] tmp = new String[7];

        String foldHead = "";
        for(int i = 1; i<folds; i++){
               foldHead += "fold"+i+";";
        }
        foldHead += "fold"+folds; // L'ultimo lo aggiungo a mano per evitare la virgola finale

        tmp[0] = "Folds";
        tmp[1] = "n";
        tmp[2] = "nu";
        tmp[3] = foldHead; // Composto da più valori separati da virgola
        tmp[4] = "AVG";
        tmp[5] = "CI L";
        tmp[6] = "CI H";
        out.add(tmp);
        return out;
    }
}