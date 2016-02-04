package edu.buaa.ml;

import java.util.HashMap;
import java.util.Map;


public class Vocabulary {
    private Map<String, Integer> wordToIndex;
    private Map<Integer, String> indexToWord;

    public Vocabulary() {
        wordToIndex = new HashMap<String, Integer>();
        indexToWord = new HashMap<Integer, String>();
    }

    public Integer getId(String word, boolean create) {
        Integer index = wordToIndex.get(word);
        if (create == false)
            return index;

        if (index == null) {
            index = wordToIndex.size();
            wordToIndex.put(word, index);
            indexToWord.put(index, word);
        }

        return index;
    }

    public Integer getId(String word) {
        return getId(word, false);
    }

    public String getWord(Integer index) {
        return indexToWord.get(index);
    }

    public int getVocabularySize() {
        return wordToIndex.size();
    }
}
