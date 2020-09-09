package classifystructuraltweets;

import java.util.ArrayList;
import java.util.List;

public class TweetsImport {
    public static List<String[]> getTweets(String[] words, String fromDate, String toDate, String position) {
        TwitterCriteria criteria = null;
        Tweet t = null;
        List<String[]> allTweets = new ArrayList<String[]>();
        for(String word : words){
            System.out.println(word);
            System.out.println(fromDate);
            criteria = TwitterCriteria.create()
                //.setUsername("barackobama")
                //.setMaxTweets(1000)
                .setQuerySearch(word)
                .setSince(fromDate)
                .setUntil(toDate)
                .setPosition(position);                    
                //.setPosition("43.7118,10.4147,3mi"); //3mi circa raggio di pisa Pisa (5km da Google Maps)                         
            for (Tweet tw : TweetManager.getTweets(criteria)) {		
                allTweets.add(new String[]{tw.getText(), tw.getDate().toString(), tw.getGeo(), word});    
            }
        }            
        return allTweets;  
    }        
}

