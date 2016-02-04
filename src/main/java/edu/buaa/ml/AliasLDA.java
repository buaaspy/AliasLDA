package edu.buaa.ml;


import java.io.*;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;


public class AliasLDA {

    // iteration number
    private int iter;


    // Metropolis-Hastings step number
    private int MHsteps;


    // M: doc number in corpus which got populated after Corpus.loadCorpus
    private int M;


    // Q: generate for each word a AliasTable
    private AliasTable[] Q;


    // V: vocabulary size
    // each word in corpus has a unique ID in vocabulary
    private int V;


    // K: assumed static topic number
    private int K;


    // z[i][j] := topic of word j in doc i
    private int[][] z;


    //
    private Vocabulary vocabulary;


    // document[i][j] := vocabulary ID for word j in doc i
    private List<int[]> documents;


    // topic-word matrix
    // nw[i][j] := # word i assigned topic j
    private int[][] nw;


    // doc-topic
    // nd[i][j] := # word assigned topic j in doc i
    private int[][] nd;


    // nwsum[j] := word number under topic j
    private int[] nwsum;


    // ndsum[i] := total topic number in doc m
    private int[] ndsum;


    // Beta: topic-word hyperparameter for Dirichlet distribution
    private double beta;
    private double vbeta;


    // Theta: doc-topic hyperparameter for Dirichlet distribution
    private double theta;


    // random := random seed
    private Random random;


    // theta: doc-topic hyperparameter
    //  beta: topic-word hyperparameter
    //     K: preassumed topic number
    public void configParameters(double theta, double beta, int K, int iter, int MHsteps) {
        this.theta = theta;
        this.beta = beta;
        this.vbeta = V * beta;
        this.K = K;
        this.iter = iter;
        this.MHsteps = MHsteps;
    }


    // assign each word in corpus a topic in random
    public void randomInitTopic() {
        nw = new int[V][K];
        nd = new int[M][K];
        nwsum = new int[K];
        ndsum = new int[M];
        z = new int[M][];
        for (int m = 0; m < M; m++) {
            int N = documents.get(m).length;
            z[m] = new int[N];
            for (int n = 0; n < N; n++) {
                int topic = random.nextInt(K);
                z[m][n] = topic;
                nw[documents.get(m)[n]][topic]++;
                nd[m][topic]++;
                nwsum[topic]++;
            }
            ndsum[m] = N;
        }

        //sunpy
        for (int k = 0; k < K; k++)
            System.out.println(String.format("nwsum[%d] = %d", k, nwsum[k]));
    }


    // construct initial Alias table for each word in each doc
    // according with current existed implementation
    // each Qw was saved without THETA
    public void initAliasTable() {
        Q = new AliasTable[V];

        for (int w = 0; w < V; w++) {
            double normalization = 0;
            double[] dist = new double[K];
            for (int k = 0; k < K; k++) {
                dist[k] = (nw[w][k] + beta) / (nwsum[k] + vbeta);
                normalization += dist[k];
            }
            for (int k = 0; k < K; k++)
                dist[k] = dist[k] / normalization;

            Q[w] = new AliasTable(dist, K, normalization);
            Q[w].generateAliasTable();
        }
    }


    public void aliasLDASampling(int m) {
        for (int n = 0; n < documents.get(m).length; n++) {

            // word unique ID in vocabulary
            int w = documents.get(m)[n];
            int topic = z[m][n];
            int oldTopic = topic;

            // remove topic-word and stat
            nw[w][topic]--;

            // sunpy
            if (nw[w][topic] < 0) {
                System.out.println("========================");
                System.out.println(String.format("word: %d", w));
                System.out.println(String.format("topic: %d", z[m][n]));
                System.out.println(String.format("original nw[%d][%d] = %d", w, topic, nw[w][topic]));
                System.out.println(String.format("nw[%d][%d] = %d", w, topic, nw[w][topic]));
                System.out.println("========================");
            }

            nwsum[topic]--;
            nd[m][topic]--;
            ndsum[m]--;

            // compute P_dw and accumulated distribution of p_dw
            double Pdw = 0.0;
            double[] p = new double[K];
            int[] sparseTopic = new int[K];
            int i = 0;
            for (int k = 0; k < K; k++) {
                if (nd[m][k] != 0) {
                    Pdw += nd[m][k] * (nw[w][k] + beta) / (nwsum[k] + vbeta);
                    p[i] = Pdw;
                    sparseTopic[i] = k;
                    i++;
                }
            }

            double prob = Pdw / (Pdw + theta * Q[w].getNormalization());

            int newTopic;
            for (int r = 0; r < MHsteps; r++) {
                if (random.nextDouble() < prob) {

                    // sample from the sparse part
                    double u = random.nextDouble() * Pdw;

                    int k;
                    for (k = 0; k < i; k++) {
                        if (u <= p[k])
                            break;
                    }

                    newTopic = sparseTopic[k == i ? (i - 1) : k];
                } else {
                    Q[w].sampleNumber++;

                    // although this part's distribution changes slowly
                    // we can reuse it for some round of iteration
                    // after that we will have to update it
                    if (Q[w].sampleNumber > K >> 1) {
                        double normalization = 0;
                        double[] dist = new double[K];
                        for (int k = 0; k < K; k++) {
                            dist[k] = k == oldTopic
                                    ? (nw[w][k] + 1 + beta) / (nwsum[k] + 1 + vbeta)
                                    : (nw[w][k] + beta) / (nwsum[k] + vbeta);

                            // sunpy
                            if (dist[k] < 0) {
                                System.out.println("====================================");
                                System.out.println(String.format("nw[%d][%d] = %d", w, k, nw[w][k]));
                                System.out.println(String.format("nwsum[%d] = %d", k, nwsum[k]));
                                System.out.println(String.format("original dist[%d] = %f", k, dist[k]));
                                System.out.println("====================================");
                            }

                            normalization += dist[k];
                        }
                        for (int k = 0; k < K; k++) {
                            dist[k] = dist[k] / normalization;
                            // sunpy
                            if (dist[k] < 0) {
                                System.out.println(String.format("dist[%d] = %f", k, dist[k]));
                                System.out.println("====================================");
                            }
                        }

                        // sample number will be cleared in construction method
                        Q[w] = new AliasTable(dist, K, normalization);
                        Q[w].generateAliasTable();
                    }

                    // sample from the dense but change slowly part
                    newTopic = Q[w].sampleAlias();
                }

                // compute the MH acceptance prob
                if (newTopic != topic) {
                    // since we change dist array in AliasTable
                    // FIXME
                    double qws = theta * Q[w].getNormalization() * Q[w].distbak[topic];
                    double qwt = theta * Q[w].getNormalization() * Q[w].distbak[newTopic];

                    double tmpOld = (nw[w][topic] + beta) / (nwsum[topic] + vbeta);
                    double tmpNew = (nw[w][newTopic] + beta) / (nwsum[newTopic] + vbeta);

                    double acceptance = ((nd[m][newTopic] +theta) / (nd[m][topic] + theta))
                            * (tmpNew / tmpOld)
                            * ((nd[m][topic] * tmpOld + qws) / (nd[m][newTopic] * tmpNew + qwt));

                    if (random.nextDouble() < acceptance)
                        topic = newTopic;
                }
            }

            // then we sampled a new Topic for current word
            z[m][n] = topic;
            nw[w][topic]++;
            nwsum[topic]++;
            nd[m][topic]++;
            ndsum[m]++;
        }
    }


    public void init() {
        Corpus corpus = new Corpus();
        try {
            corpus.loadCorpus("data/mini");
            System.out.println("vocabulary size: " + corpus.getVocabularySize());
            System.out.println("doc number: " + corpus.getDocNumber());
        } catch (IOException e) {
            System.out.println("exception occured when loading corpus !!! detailed msg: " + e.toString());
        }

        this.documents = corpus.documents;
        this.vocabulary = corpus.vocabulary;
        this.V = corpus.getVocabularySize();
        this.M = corpus.getDocNumber();
        random = new Random();

        configParameters(Constants.THETA, Constants.BETA, Constants.TOPIC, Constants.ITER, Constants.MHSTEP);
        randomInitTopic();
        initAliasTable();
    }


    public void train() {
        for (int i = 0; i < iter; i++) {
            System.out.print(".");
            for (int m = 0; m < M; m++) {
                aliasLDASampling(m);
            }
        }
    }


    class TopicWord {
        public int word;
        public int count;
    }


    class comparator implements Comparator {

        public int compare(Object o1, Object o2) {
            TopicWord l = (TopicWord)o1;
            TopicWord r = (TopicWord)o2;

            return l.count == r.count ? 0 : (l.count > r.count ? -1 : 1);
        }
    }


    public void showTopicWord() throws IOException {
        File out = new File("model/topicword.dat");
        FileOutputStream outputStream = new FileOutputStream(out);
        BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(outputStream);

        for (int k = 0; k < K; k++) {
            bufferedOutputStream.write(String.format("topic: %d: ", k).getBytes());
            TopicWord[] wordlist = new TopicWord[V];
            for (int w = 0; w < V; w++) {
                wordlist[w] = new TopicWord();
                wordlist[w].count = nw[w][k];
                wordlist[w].word = w;
            }
            Arrays.sort(wordlist, new comparator());

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Constants.WORD_NUM_UNDER_TOPIC; i++) {
                sb.append(String.format("%s(%f)(%d, %d)\t",
                        vocabulary.getWord(wordlist[i].word),
                        ((double)wordlist[i].count / nwsum[k]),
                        wordlist[i].count, nwsum[k]));
            }
            sb.append("\n");

            bufferedOutputStream.write(sb.toString().getBytes());
        }

        bufferedOutputStream.flush();
        outputStream.close();
        bufferedOutputStream.close();
    }


    // test case
    public static void TestAliasSampler() {
        double[] dist = new double[]{0.1, 0.2, 0.3, 0.3};
        double normalization = 0;
        int number = dist.length;

        for (int i = 0; i < dist.length; i++)
            normalization += dist[i];

        AliasTable aliasMethod = new AliasTable(dist, number, normalization);
        aliasMethod.generateAliasTable();

        int[] equips = new int[]{0, 0, 0, 0};
        int runnum = 100000;
        int equip;
        for (int i = 0; i < runnum; i++) {
            equip = aliasMethod.sampleAlias();
            equips[equip]++;
        }

        System.out.printf("weapon = %d, deco = %d, ring = %d, cape = %d\n",
                equips[0],
                equips[1],
                equips[2],
                equips[3]);
        System.out.printf("weapon = %.2f, deco = %.2f, ring = %.2f, cape = %.2f\n",
                (double)(equips[0] * 100) / runnum,
                (double)(equips[1] * 100) / runnum,
                (double)(equips[2] * 100) / runnum,
                (double)(equips[3] * 100) / runnum);
    }


    public static void TestCorpus() throws IOException {
        Corpus corpus = new Corpus();
        corpus.loadCorpus("data/mini");

        System.out.println("vocabulary size: " + corpus.getVocabularySize());
        System.out.println("doc number: " + corpus.getDocNumber());

        corpus.printPartOfVocabulary(100);
    }


    public void TestSort() {
        TopicWord[] tw = new TopicWord[4];
        for (int i = 0; i < 4; i++) {
            tw[i] = new TopicWord();
            tw[i].count = i;
            tw[i].word = i;
        }

        Arrays.sort(tw, new comparator());
        for (int i = 0; i < 4; i++)
            System.out.println(String.format("tw[%d].count = %d", i, tw[i].count));
    }


    public static void main(String[] args) {
        AliasLDA aliasLDA = new AliasLDA();
        aliasLDA.init();
        System.out.println("load corpus done !");
        aliasLDA.train();
        System.out.println("train done !");

        try {
            aliasLDA.showTopicWord();
            System.out.println("write topic-word done !");
        } catch (IOException e) {
            System.out.println("exception occured when show topic word, detailed information: " + e.toString());
        }
    }
}
