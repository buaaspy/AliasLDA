package edu.buaa.ml;


public class Constants {
    public static final int TOPIC = 10;
    public static final double THETA = 20 / TOPIC;
    public static final double BETA = 0.5;
    public static final int ITER = 1000;
    public static final int INFERENCE_ITER = 50;
    public static final int MHSTEP = 2;
    public static final int WORD_NUM_UNDER_TOPIC = 10;
    public static final String COMMENT_PREFIX = "#";
    public static final String MODEL_PARAMS_SAVED_PATH = "model/modelparams.dat";
    public static final String MODEL_PHIS_SAVED_PATH = "model/phis.dat";
}
