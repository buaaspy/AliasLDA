package edu.buaa.ml;

import java.io.*;
import java.util.LinkedList;
import java.util.List;


public class Corpus {
    public Vocabulary vocabulary;
    public List<int[]> documents;

    public Corpus() {
        documents = new LinkedList<int[]>();
        vocabulary = new Vocabulary();
    }

    public int[] rawDoc2Indexes(List<String> rawDoc) {
        int[] wordIndexes = new int[rawDoc.size()];
        int i = 0;
        for (String word : rawDoc)
            wordIndexes[i++] = vocabulary.getId(word, true);

        return wordIndexes;
    }

    public void addDocument(int[] docWordIndexes) {
        documents.add(docWordIndexes);
    }

    public void loadCorpus(String folderPath) throws IOException {
        File folder = new File(folderPath);
        for (File file : folder.listFiles()) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            List<String> docStringList = new LinkedList<String>();
            String line;
            while ((line = reader.readLine()) != null) {
                String words[] = line.split(" ");
                for (String word : words) {
                    if (word.trim().length() <= 1)
                        continue;
                    docStringList.add(word.trim());
                }
            }
            reader.close();
            addDocument(rawDoc2Indexes(docStringList));
        }
    }

    public int getDocNumber() {
        return documents.size();
    }

    public int getVocabularySize() {
        return vocabulary.getVocabularySize();
    }

    public void printPartOfVocabulary(int n) {
        for (int i = 1; i <= n; i++) {
            System.out.print("( " + i + ", " + vocabulary.getWord(i - 1) + ")\t");
            if (i % 4 == 0 && i != 0)
                System.out.print("\n");
        }
    }

    public String getVocabularyWord(int index) {
        return vocabulary.getWord(index);
    }
}
