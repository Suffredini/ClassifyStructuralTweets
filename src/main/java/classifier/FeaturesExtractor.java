package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import static java.util.stream.Collectors.toMap;

public class FeaturesExtractor {
    
    public String[] deleteUrlAndPic(String[] tweets){
        String urlRegex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+";
        String picRegex = "pic.twitter.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+";
        for(int i = 0; i<tweets.length; i++){
            tweets[i] = tweets[i].replaceAll(urlRegex, "").replaceAll(picRegex, "").replaceAll("[^a-zA-Z ]", " ").toLowerCase();
        }  
        return tweets;
    }
    
    public List<List<String>> tokenization(String[] tweets){
        List<List<String>> tweetsTokenized = new ArrayList<>();
        List<String> allWord;
        for(String tweet : tweets){
            String[] words = tweet.split(" ");
            allWord = new ArrayList<>();
            for(String word : words){
                if(!word.isEmpty()){
                    allWord.add(word);
                }
            }
            tweetsTokenized.add(allWord);
        }
        return tweetsTokenized;
    }
    
    public List<List<String>> stopWordFiltering(List<String> stopWords, List<List<String>> tokenized){
        List<List<String>> tweetsNoStopWord = new ArrayList<>();
        for(List<String> tweet : tokenized){
            tweet.removeAll(stopWords);
            tweetsNoStopWord.add(tweet);
        }
        return  tweetsNoStopWord;
    }
    
    public List<List<String>> stemming(List<String[]> stemmingList, List<List<String>> tweetsNoStopWord){
        List<List<String>> tweetsStemmed = new ArrayList<>();
        HashMap<String,String> stemmingWords = new HashMap<>();
        for(String[] word : stemmingList){
            stemmingWords.put(word[0], word[1]);
        }
        String newWord;
        List<String> newTweet;
        for(List<String> tweet : tweetsNoStopWord){
            newTweet = new ArrayList<>();
            for(String word : tweet){
                newWord = stemmingWords.get(word);
                if(newWord != null){                    
                    word = newWord;
                }
                newTweet.add(word);
            }
            tweetsStemmed.add(newTweet);
        }
        return  tweetsStemmed;
    }
    
    // Prendo le prime f features dopo averle ordinate per peso
    public Map<String, Double> computeSumWeight(List<List<String>> tweetsStemmed,  int numberTweets) {
     //SUPERVISED LEARNING STAGE
        //Estraggo tutte le Stem word da tutte le SUM contando in quante Sum compare la Stem word
        HashMap<String,Double> cstr = new HashMap<>();
        List<String> countedStem = new ArrayList<>();
        Double value;
        for(List<String> sum : tweetsStemmed){
            for(String stem : sum){
                //Verifico se ho già incontrato la stem in questa SUM
                if(countedStem.contains(stem))
                    continue;
                
                if(cstr.containsKey(stem)){
                    value = cstr.get(stem);
                    cstr.replace(stem, ++value);
                } else {
                    cstr.put(stem, new Double(1));
                }                
                countedStem.add(stem);
            }   
            countedStem.clear();
        }

        //PESO DI OGNIO STEM MEDIANTE IDF
        HashMap<String, Double>  stemWeight = new HashMap<>();
        for(Map.Entry<String, Double> entry: cstr.entrySet()) {
            //TODO sostituire numberTweets con dimensione del training set
            stemWeight.put(entry.getKey(),Math.log(numberTweets/entry.getValue()));
        }
        
        //STOP PESO DI OGNIO STEM MEDIANTE IDF
        /*
        //Riordino l'HashMap Decrescente per Valore
        Map<String, Double>  cstrSorted = stemWeight
            .entrySet()
            .stream()
            .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
            .collect(
                toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2,
                    LinkedHashMap::new));
        //Stampo HashMap
        /*for(Entry<String, Double> entry: cstrSorted.entrySet()) {
            System.out.println(entry.getKey() + " : " + entry.getValue());
        }*/
        
        return stemWeight;    
    }
    
    public List<String[]> completeSetOfStems(List<List<String>> tweetsStemmed, Map<String, Double> weightedSum, String[] classes){
        List<String[]> cs = new ArrayList<>();
        int nTweet = 0;
        int pos;
        String[] bag;
        int nAttribute = weightedSum.size();
        // Inserisco l'intestazione
        bag = new String[nAttribute+1];
        pos=0;
        for(Map.Entry<String, Double> word: weightedSum.entrySet()) {
           bag[pos++] = word.getKey();   
        }
        bag[nAttribute] = "classAttribute";
        cs.add(bag);
        
        // Inserisco gli attributi
        for(List<String> sum : tweetsStemmed){
            pos = 0;
            bag = new String[nAttribute+1];           
            for(Map.Entry<String, Double> word: weightedSum.entrySet()) {
                if(sum.contains(word.getKey())){
                    bag[pos] = word.getValue().toString();
                } else {
                    bag[pos] = Integer.toString(0);
                }                
                pos++;
            }
            bag[nAttribute] = getClass(classes[nTweet++]);
            cs.add(bag);
        }
        return  cs;
    }
    
      
    private static String getClass(String classe) {
        /* distinsione poco/molto grave
        switch(classe){
            case "10000":
                return "NS";
            case "01000":
                return "PG";
            case "00100":
                return "PG";
            case "00010":
                return "MG";
            case "00001":
                return "MG";
        }
        return "NS";
        */
        switch(classe){
            case "10000":
                return "NS";
            case "01000":
                return "S"; //SN
            case "00100":
                return "NS"; //MN
            case "00010":
                return "S"; //SG
            case "00001":
                return "NS"; //MG
        }
        return "NS";
    }

    String[] extractAllAttributes(List<List<String>> tweetsStemmed) {
      List<String> allAttributes = new ArrayList<>();
        for(List<String> sum : tweetsStemmed){
            for(String stem : sum){
                if(allAttributes.contains(stem))
                    continue;         
                allAttributes.add(stem);
            } 
        }   
        allAttributes.add("classAttribute");
        String[] ret = new String[allAttributes.size()];
        int pos = 0;
        for(String word : allAttributes){
            ret[pos++] = word;
        }        
        return ret;
    }

    List<String[]> getArrayAllTweets(String[] allAttributes, String[] classes, List<List<String>> tweetsStemmed) {
        List<String[]> arrayAllTweets = new ArrayList<>();
        int nTweet = 0;
        int pos;
        String[] bag;
        // Inserisco l'intestazione
        arrayAllTweets.add(allAttributes);
        
        // Inserisco gli attributi
        for(List<String> sum : tweetsStemmed){
            bag = new String[allAttributes.length];           
            for(pos=0; pos<allAttributes.length-1; pos++) {
                if(sum.contains(allAttributes[pos])){
                    bag[pos] = "1";
                } else {
                    bag[pos] = "0";
                }                
            }            
            bag[allAttributes.length-1] = getClass(classes[nTweet++]);
            arrayAllTweets.add(bag);
        }
        return  arrayAllTweets;
    }
    
    
}
