/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import classifystructuraltweets.TweetsImport;
import java.text.ParseException;
import java.util.List;


/**
 *
 * @author suffr
 */
public class DownloadTweets {
     public static void main(String[] args) throws ParseException{
        String[] keyWord = {"guasto"};
                            /*"condotte","condotta", "pioggia", "condotto","scarico",
                                "scarichi", "scarica", "scoli", "scolo", "scola",
                                "sommersa", "sommerge", "sommergere"*/
        String[] key = new String[1];
        List<String[]> tweets;
        for(String k:keyWord){            
            key[0] = k;
            tweets = TweetsImport.getTweets(key, "2000-01-01", "2020-03-01", "43.7118,10.4147,10000mi");
            System.out.println("\n\n"+k+" ("+tweets.size()+")\n");
            for(String[] s : tweets){
                System.out.println(s[0]);
            }
            System.out.println("\n \n");
        }   
         
     }
}
