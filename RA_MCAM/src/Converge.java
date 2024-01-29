import jdk.nashorn.internal.objects.Global;
import org.w3c.dom.Attr;

import java.io.*;
import java.util.*;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class Converge {

    //全局属性：用于转换属性值
    static List<Map<String, Integer>> convert = new ArrayList<>();

    static List<Map<Integer, String>> recoverDicts = new ArrayList<>();

    //记录数据中所有维度包含属性的数量
    public static int[] globalSize;

    static PriorityQueue<Tensor> tensorQueue = new PriorityQueue<>((o1, o2) -> {
        double a = Util.computeDensity(o1);
        double b = Util.computeDensity(o2);
        return Double.compare(a, b);
    });

    public static int globalMaxLatitude = Integer.MIN_VALUE;

    public static int globalMinLatitude = Integer.MAX_VALUE;

    public static int globalMaxlongitude = Integer.MIN_VALUE;

    public static int globalMinlongitude = Integer.MAX_VALUE;

    public static int globalMaxTime = Integer.MIN_VALUE;

    public static int globalMinTime = Integer.MAX_VALUE;

    public static int[][] attVals; //保存每一条记录每个维度的信息

    //全局张量
    static Tensor tensor;
    // static double TEMPORAL_THRESHOLD;
    // static double SPATIAL_THRESHOLD;
    static int CORE_COLUMN_LENGTH;
    static int TIME_COLUMN;
    static int LATITUDE_COLUMN;
    static int LONGITUDE_COLUMN;

    static int DENSITY_PROPORTION = 1;
    static int TEMPORAL_PROPORTION = 1;
    static int SPATIAL_PROPORTION = 1;

    static double timePropogation = 100;
    static double spacePropogation = 100;

    static long demoTime1 = 0;
    static long demoTime2 = 0;
    static long demoTime3 = 0;
    static long demoTime4 = 0;


    /**
     * 输入路径input_path
     * 输出路径 output_path
     * 分隔符delim，默认为","
     * 查找子张量个数k，默认k = 0
     * CORE_PROPORTION
     * TIME_PROPORTION
     * SPACE_PROPORTION
     * @param args ： 0为输入路径，1为输出路径，2为分隔符，3为寻找子张量个数
     */
    public static void main(String[] args) throws Exception {
        if(args.length < 4){
            System.err.println("please enter parameters in the following format: ");
            System.err.println("input_path output_path splitStr k(Number of sub-tensor to search) Density_proportion Temporal_proportion Spatial_proportion.");
            return;
        }
        final String input_path = args[0];
        final String output_path = args[1];
        final String delim = args[2];
        int k = Integer.parseInt(args[3]);

        if (args.length >= 7) {
            DENSITY_PROPORTION = Integer.parseInt(args[4]);
            TEMPORAL_PROPORTION = Integer.parseInt(args[5]);
            SPATIAL_PROPORTION = Integer.parseInt(args[6]);
        }
        //有时空约束使用时空约束，否则默认没有约束
        if(args.length >= 9){
            timePropogation = Double.parseDouble(args[7]);
            spacePropogation = Double.parseDouble(args[8]);
        }
        System.out.println("input_path: " + input_path);
        System.out.println("output_path: " + output_path);
        System.out.println("splitStr: \"" + delim + "\"");
        System.out.println("k (Number of sub-tensor to search): " + k + " (Default: 0)");

        long start = System.currentTimeMillis();

        System.out.println("Converting data into tensors...");

        //arr前三个值为各个维度的大小，最后一个值为记录的条数
        long startTime = System.currentTimeMillis();
        preComputeParameter(input_path, delim);
        long endTime = System.currentTimeMillis();
        System.out.println("查找各个维度大小信息的时间为："+ (endTime - startTime) + "ms");

        startTime = System.currentTimeMillis();
        tensor = createTensor(input_path, delim);
        endTime = System.currentTimeMillis();
        System.out.println("构造父张量的时间为："+ (endTime - startTime) + "ms");
        System.out.println("demoTime3耗时为：" + demoTime3 + "ms");

        //调用findSubTensor()函数获取到k个子张量信息，该信息使用SubTensor对象数组保存。
        List<Tensor> subTensors = findSubTensors(k);
        System.out.println("Running time: " + (System.currentTimeMillis() - start + 0.0) / 1000 + " seconds");

        //得到k个子张量信息之后，调用outputTensorList函数将其输出
        System.out.println("A total of " + subTensors.size() + " dense sub-tensors were found.");
        System.out.println("Writing outputs...\n");
        long startTime2 = System.currentTimeMillis();
        outputTensorList(subTensors, output_path, subTensors.size());
        System.out.println("输出子张量时间为：" + (System.currentTimeMillis() - startTime2) + "ms");
        System.out.println("Outputs were written: " + (System.currentTimeMillis() - start + 0.0) / 1000 + " seconds was taken.");
    }

    /**
     * 计算全局参数
     * @param input_path
     * @param splitStr
     * @throws Exception
     */
    private static void preComputeParameter(String input_path, String splitStr) throws Exception {
        globalSize = Util.computerSize(input_path,splitStr);
        CORE_COLUMN_LENGTH = globalSize.length - 4;
        TIME_COLUMN = globalSize.length - 4;
        LATITUDE_COLUMN = globalSize.length - 3;
        LONGITUDE_COLUMN = globalSize.length - 2;
    }

    /**
     * 找到指定k个密集子张量
     *
     * @param k 要求返回的子张量个数
     * @return 子张量数组
     */
    private static List<Tensor> findSubTensors(int k) throws CloneNotSupportedException {
        PriorityQueue<Pair<Tensor, Double>> subTensorPriorityQueue = new PriorityQueue<>(k, (o1, o2) -> {
            double a = Util.computeDensity(o1.getKey());
            double b = Util.computeDensity(o2.getKey());
            return Double.compare(a, b);
        });

        //设置迭代计数器，每10次操作输出一次
        int countNum = 1;

        //TODO：是否进行循环
        while(subTensorPriorityQueue.size() != k){
            List<Integer> globalList = new LinkedList<>();
            int[] value = tensor.getValue();
            for (int i = 0; i < value.length; i++){
                if(value[i] != 0){
                    globalList.add(i);
                }
            }
            long start = System.currentTimeMillis();
            findSubTensor(globalList, subTensorPriorityQueue, k, 0);
            demoTime1 += System.currentTimeMillis() - start;
            countNum++;
            if(countNum % 10 == 0){
                System.out.println("当前迭代执行次数为：" + countNum + "次");
                System.out.println("当前队列中待处理tensor个数：" + tensorQueue.size() + "个");
                System.out.println("当前密集子张量队列子张量个数：" + subTensorPriorityQueue.size() + "个");
                System.out.println("查找密集子张量的时间：" + demoTime1 + "ms");
                System.out.println("花费在创建上的时间：" + demoTime4 + "ms" );
                System.out.println("demoTime3为：" + demoTime3 + "ms");
                System.out.println("花费在Mbiz上的时间为：" + demoTime2 + "ms");
                System.out.println();
            }
        }

        List<Tensor> subTensors = new ArrayList<>();
        for (int i = 0; i < k; i++){
            if(subTensorPriorityQueue.peek() == null)break;
            Tensor subTensor = subTensorPriorityQueue.poll().getKey();
            subTensors.add(subTensor);
        }
        Collections.reverse(subTensors);

        return subTensors;
    }

    /**
     * 对tensor中的切片进行整理
     *
     * @return ：返回按照质量排好序的属性值列表，其中0为属性值，1为该属性的质量
     */
    private static int[][] orderSliceByDensity(Tensor tensor, int mode) {
        int size = tensor.getSize()[mode];
        int[][] sliceInformation = new int[2][size];
        List<List<Slice>> slices = tensor.getSlices();
        int[] value = tensor.getValue();

        // 计算出每一个切片的质量，按照质量对切片进行排序
        // 将time、longitude、latitude维度的切片统一到一个列表中，加上维度信息
        int index = 0;
        int existSize = 0;
        List<Slice> sliceList = slices.get(mode);
        for (Slice s : sliceList) {
            if (s.isExists()) {
                existSize++;
                sliceInformation[0][index] = s.getIndex();
                // 计算切片质量
                int tempMass = 0;
                Vector<Integer> dataIndexs = s.getDataIndexs();
                for (Integer data : dataIndexs) {
                    if (value[data] != 0) tempMass += value[data];
                }
                sliceInformation[1][index++] = tempMass;
            }
        }

        sort(sliceInformation, 0, existSize - 1);

        return sliceInformation;
    }

    public static void findSubTensor(List<Integer> globalList, PriorityQueue<Pair<Tensor, Double>> subTensorPriorityQueue, int k, int circleNum) throws CloneNotSupportedException {
        while(true){
            double measurementBefore = Util.computeMeasurementByList(globalList);
            //选择一个cardinality最大的维度
            int[] subTensorValue = tensor.getValue();
            //统计每个时空属性的mass
            int[][] stMass = new int[3][];
            stMass[0] = new int[globalMaxTime + 1];
            stMass[1] = new int[globalMaxLatitude + 1];
            stMass[2] = new int[globalMaxlongitude + 1];
            for (Integer integer : globalList) {
                stMass[0][attVals[TIME_COLUMN][integer]] += subTensorValue[integer];
                stMass[1][attVals[LATITUDE_COLUMN][integer]] += subTensorValue[integer];
                stMass[2][attVals[LONGITUDE_COLUMN][integer]] += subTensorValue[integer];
            }

            int[] size = new int[3];
            for(int i = 0; i < 3; i++){
                int len = stMass[i].length;
                for(int j = 0; j < len; j++){
                    if(stMass[i][j] != 0)size[i]++;
                }
            }

            //不允许太小
            if (size[0] < globalSize[TIME_COLUMN] * 0.2 || (double)size[1] < globalSize[LATITUDE_COLUMN] * 0.2 || size[2] < globalSize[LONGITUDE_COLUMN] * 0.2) {
                break;
            }

            int dimension = -1;
            int maxSize = 0;
            for(int i = 0; i < 3; i++){
                if(size[i] > maxSize){
                    maxSize = size[i];
                    dimension = i;
                }
            }

            int sum = Arrays.stream(stMass[dimension]).sum();
            double avgMass = sum * 1.0 / maxSize;

            Set<Integer> deleteSlices = new HashSet<>();
            for(int i = 0; i < stMass[dimension].length; i++){
                if(stMass[dimension][i] != 0 && stMass[dimension][i] < avgMass){
                    deleteSlices.add(i);
                }
            }
            globalList = Util.deleteSlices(deleteSlices, TIME_COLUMN + dimension, globalList);
            double measurementAfter = Util.computeMeasurementByListNotSet(globalList);
            if(measurementAfter <= measurementBefore){
                break;
            }
        }

        //创建subTensor
        Tensor subTensor1 = tensor.createSubTensorBydataList(globalList);

        processCoreDimension(subTensor1, k, subTensorPriorityQueue, circleNum);
    }




    /**
     * 整合处理子张量的操作，包括对其它两个维度的贪心，判断是否为前k个密集子张量，以及从tensor中去除
     * @param subTensor ： 当前子张量信息
     * @param k ： 查找的子张量个数
     * @param subTensorPriorityQueue ： 存储查找到的密集子张量的优先队列
     */
    public static void processCoreDimension(Tensor subTensor, int k, PriorityQueue<Pair<Tensor, Double>> subTensorPriorityQueue, int circleNum) throws CloneNotSupportedException {
        double density1 = Util.computeDensity(subTensor);
        // double density2 = subTensorPriorityQueue.peek() == null ? 0.0 : subTensorPriorityQueue.peek().getValue();
        //第二轮后初始重量小于队列中最小子张量初始mass的不进行update操作，降低时间复杂度
        //第一轮中初始mass不到队列中最小张量一半的不进行update操作
        // if(subTensorPriorityQueue.size() == k && (circleNum > 0 && density1 <= density2 || circleNum == 0 && density1 <= density2 / 2)){
        //     updateTensorByDeleteSubTensor(subTensor);
        //     return;
        // }

        long startTime = System.currentTimeMillis();
        // updateCoreDimensionLikeDcube(subTensor);
        // updateCoreDimensionLikeMzoom(subTensor);
        updateCoreDimensionLikeMbiz(subTensor);
        demoTime2 += System.currentTimeMillis() - startTime;
        //TODO:切换度量时需要修改此处
        double timeRange = Util.computeTemporalRange(subTensor.getMinTime(), subTensor.getMaxTime()) * 100;
        double spaceRange = Util.computeSpatialRange(subTensor.getMinLatitude(), subTensor.getMaxLatitude(), subTensor.getMinLongitude(), subTensor.getMaxLongitude()) * 100;
        if(timeRange <= timePropogation && spaceRange <= spacePropogation){
            if(subTensorPriorityQueue.size() >= k){
                Tensor minTensor = subTensorPriorityQueue.peek() == null ? null :subTensorPriorityQueue.peek().getKey();
                if(minTensor != null && Util.computeDensity(minTensor) < Util.computeDensity(subTensor)){
                    subTensorPriorityQueue.poll();
                    subTensorPriorityQueue.offer(new Pair<>(subTensor,density1));
                }
            }else{
                subTensorPriorityQueue.offer(new Pair<>(subTensor,density1));
            }
        }

        //将子张量数组中所有子数组从全局张量tensor中去除
        updateTensorByDeleteSubTensor(subTensor);

    }

    /**
     * 从全局张量中删除找到的子张量
     * 此处删除不更新全局tensor的最大最小经纬度和时间信息
     * @param subTensor ：需要从全局张量tensor中删除的子张量
     */
    private static void updateTensorByDeleteSubTensor(Tensor subTensor) {
        //更新质量
        tensor.setMass(tensor.getMass() - subTensor.getMass());

        //遍历所有数据
        int[] subValue = subTensor.getValue();
        int[] value = tensor.getValue();
        int[] size = tensor.getSize();
        List<List<Slice>> slices = tensor.getSlices();
        List<Map<Integer, Integer>> dataOfAttr = tensor.getDataOfAttr();
        for (int i = 0; i < subValue.length; i++){
            if(subValue[i] != 0){
                //修改user、item以及time纬度
                for(int j = 0; j < Converge.globalSize.length - 1; j++){
                    int attr = attVals[j][i];
                    Map<Integer, Integer> dataOfAttrInMode = dataOfAttr.get(j);
                    if(dataOfAttrInMode.containsKey(attr)){
                        dataOfAttrInMode.replace(attr, dataOfAttrInMode.get(attr) - value[i]);
                        if(dataOfAttrInMode.get(attr) == 0){
                            dataOfAttrInMode.remove(attr);
                            //将对应的切片设置为不存在
                            slices.get(j).get(attr - 1).setExists(false);
                            size[j]--;
                        }
                    }
                }

                //删除该tuple
                value[i] = 0;

            }
        }
    }

    //Mzoom删除核心维度的子张量时，假设不会同时影响到时空维度
    private static void updateCoreDimensionLikeMzoom(Tensor subTensor){
        //创建一个用于遍历删除的压缩版subTensor
        Tensor tempTensor = new Tensor(subTensor);
        //创建小顶堆
        final IMinHeap[] heaps = createHeaps(subTensor);
        int mass = tempTensor.getMass();
        int[] cardinality = tempTensor.getSize().clone();
        int sumOfCardinality = 0;
        for(int i = 0; i < CORE_COLUMN_LENGTH; i++){
            sumOfCardinality += cardinality[i];
        }
        List<Map<Integer, Integer>> dataOfAttr = tempTensor.getDataOfAttr();

        int delIndex = 0;
        int[] deleteModes = new int[sumOfCardinality];
        int[] deleteAttrs = new int[sumOfCardinality];

        int maxIters = 0;
        double maxDensityAmongIters = Util.computeDensity2(tempTensor);
        int i = 0;

        while(i < sumOfCardinality){
            int maxMode = 0;
            double maxDensityAmongModes = -Double.MAX_VALUE;
            for(int mode = 0; mode < Converge.CORE_COLUMN_LENGTH; mode++){
                final Pair<Integer, Integer> pair = heaps[mode].peek();
                if(pair != null){
                    double tempDensity = Util.computeDensity(mass - pair.getValue(), Util.computeVolume(tempTensor) - 1);
                    if(tempDensity >= maxDensityAmongModes){
                        maxMode = mode;
                        maxDensityAmongModes = tempDensity;
                    }
                }
            }
            Pair<Integer, Integer> pair = heaps[maxMode].poll();

            mass -= pair.getValue();
            double density = Util.computeDensity(tempTensor.getMass() - pair.getValue(), Util.computeVolume(tempTensor) - 1);
            //后续需要更新tempTensor中的Mass和Size
            tempTensor.setMass(tempTensor.getMass() - pair.getValue());
            int[] tempSize = tempTensor.getSize();
            tempSize[maxMode]--;
            tempTensor.setSize(tempSize);

            if(density > maxDensityAmongIters){
                maxDensityAmongIters = density;
                maxIters = i + 1;
            }
            deleteModes[delIndex] = maxMode;
            deleteAttrs[delIndex++] = pair.getKey();
            i++;

            removeAndUpdateAttValMassesLikeMzoom(tempTensor, maxMode, pair.getKey(), dataOfAttr, heaps);
        }

        //真正的子张量删除阶段
        for(int index = 0; index < maxIters; index++){
            if(deleteAttrs[index] != 0){
                subTensor.deleteAttr(deleteAttrs[index], deleteModes[index]);
            }
        }
    }

    static class AttrComparator implements Comparator<Integer>{

        int[] value;
        double[][] spaceVals;

        public AttrComparator(int[] value, double[][] spaceVals) {
            this.value = value;
            this.spaceVals = spaceVals;
        }

        @Override
        public int compare(Integer o1, Integer o2) {
            if(value[o1] == 0 && value[o2] != 0){
                return -1;
            }else if(value[o1] != 0 && value[o2] == 0){
                return 1;
            }else if(value[o1] == 0 && value[o2] == 0){
                return 0;
            }else{
                return -Double.compare(spaceVals[1][o1], spaceVals[1][o2]);
            }
        }
    }

    private static void updateCoreDimensionLikeMbiz(Tensor subTensor) throws CloneNotSupportedException {
        updateCoreDimensionLikeMzoom(subTensor);
        // Tensor globalTensor = SerializationUtils.clone(subTensor);
        // List<List<Slice>> globalSlices = globalTensor.getSlices();
        int[] globalValue = subTensor.getValue().clone();
        List<List<Slice>> globalSlices = new ArrayList<>();
        List<List<Slice>> initSlices = subTensor.getSlices();
        for (List<Slice> sliceList : initSlices) {
            List<Slice> sliceListCopy = new ArrayList<>();
            for (Slice s: sliceList) {
                sliceListCopy.add(s.clone());
            }
            globalSlices.add(sliceListCopy);
        }

        final byte[] nonmemberCounts = new byte[globalValue.length];
        List<List<Slice>> slices = subTensor.getSlices();

        int[][] attributeToValueToMassChange = new int[CORE_COLUMN_LENGTH][];
        for (int mode = 0; mode < CORE_COLUMN_LENGTH; mode++) {
            if(mode < TIME_COLUMN){
                attributeToValueToMassChange[mode] = new int[globalSize[mode] + 1];
            }else{
                attributeToValueToMassChange[mode] = new int[globalMaxTime + 1];
            }
        }

        for(int i=0; i< globalValue.length; i++){
            int measureValue = globalValue[i];
            if(measureValue == 0)continue;

            //记录有多少tuple已经被删除了
            byte nonmemberCount = 0;
            int nonmemberMode = 0;
            for(int mode=0; mode<CORE_COLUMN_LENGTH; mode++){
                if(!slices.get(mode).get(attVals[mode][i] - 1).isExists()) {
                    nonmemberCount += 1;
                    nonmemberMode = mode;
                }
            }

            nonmemberCounts[i] = nonmemberCount;

            if(nonmemberCount==0) { // in the block
                for(int mode=0; mode<CORE_COLUMN_LENGTH; mode++){
                    attributeToValueToMassChange[mode][attVals[mode][i]] += measureValue;
                }
            }
            else if(nonmemberCount == 1){
                attributeToValueToMassChange[nonmemberMode][attVals[nonmemberMode][i]] += measureValue;
            }
        }

        final IMinHeap[] inHeaps = new IMinHeap[CORE_COLUMN_LENGTH];
        final IMaxHeap[] outHeaps = new IMaxHeap[CORE_COLUMN_LENGTH];
        final int[] attributeToCardinalities = new int[CORE_COLUMN_LENGTH];
        int sumOfCardinalities = 0;

        //初始化大小顶堆
        for(int mode = 0; mode < CORE_COLUMN_LENGTH; mode++) {
            List<Slice> sliceList = slices.get(mode);
            int[] attValToMassChange = attributeToValueToMassChange[mode];
            IMinHeap inHeap = null;
            IMaxHeap outHeap = null;
            if(mode < TIME_COLUMN){
                inHeap = new HashIndexedMinHeap(globalSize[mode] + 1);
                outHeap = new HashIndexedMaxHeap(globalSize[mode] + 1);
                for(int index = 1; index < globalSize[mode] + 1; index++) {
                    if(sliceList.get(index - 1).isExists()) {
                        inHeap.insert(index, attValToMassChange[index]);
                        attributeToCardinalities[mode] += 1;
                        sumOfCardinalities ++;
                    }
                    else {
                        outHeap.insert(index, attValToMassChange[index]);
                    }
                }
            }else{
                inHeap = new HashIndexedMinHeap(globalMaxTime + 1);
                outHeap = new HashIndexedMaxHeap(globalMaxTime + 1);
                for(int index = 1; index < globalMaxTime + 1; index++) {
                    if(sliceList.get(index - 1).isExists()) {
                        inHeap.insert(index, attValToMassChange[index]);
                        attributeToCardinalities[mode] += 1;
                        sumOfCardinalities ++;
                    }
                    else {
                        outHeap.insert(index, attValToMassChange[index]);
                    }
                }
            }

            inHeaps[mode] = inHeap;
            outHeaps[mode] = outHeap;
        }

        double currentScore = Util.computeDensity(subTensor);
        while(true) {

            double previousScore = currentScore;
            int maxMode = 0;
            boolean action = false; //false: remove, true: insert
            // if(sumOfCardinalities > lower) {
            if(sumOfCardinalities > 0) {
                for (int mode = 0; mode < CORE_COLUMN_LENGTH; mode++) {
                    final Pair<Integer, Integer> pair = inHeaps[mode].peek();
                    if (pair != null) {
                        double tempScore = Util.computeDensity(subTensor.getMass() - pair.getValue(), Util.computeVolume(subTensor) - 1);
                        // double tempScore = measure.ifRemoved(attribute, 1, pair.getValue());
                        if (tempScore > currentScore) {
                            maxMode = mode;
                            action = false;
                            currentScore = tempScore;
                        }
                    }
                }
            }

            if(sumOfCardinalities < Integer.MAX_VALUE) {
                for (int mode = 0; mode < CORE_COLUMN_LENGTH; mode++) {
                    final Pair<Integer, Integer> pair = outHeaps[mode].peek();
                    if (pair != null) {
                        double tempScore = Util.computeDensity(subTensor.getMass() + pair.getValue(), Util.computeVolume(subTensor) + 1);
                        // double tempScore = measure.ifInserted(attribute, 1, pair.getValue());
                        if (tempScore > currentScore) {
                            maxMode = mode;
                            action = true;
                            currentScore = tempScore;
                        }
                    }
                }
            }

            if(currentScore == previousScore) { // terminates
                break;
            }
            int[] values = subTensor.getValue();
            List<Map<Integer, Integer>> dataOfAttr = subTensor.getDataOfAttr();

            int[] size = subTensor.getSize();

            if(action == false) { //remove
                Pair<Integer, Integer> pair = inHeaps[maxMode].poll();
                int attValToRemove = pair.getKey();
                //更新mass
                subTensor.setMass(subTensor.getMass() - pair.getValue());

                sumOfCardinalities--;
                attributeToCardinalities[maxMode]--;

                //update degree in
                int massSumOut = 0;
                List<Integer> tuples = globalSlices.get(maxMode).get(attValToRemove - 1).getDataIndexs();
                for (int tuple : tuples) {
                    if(globalValue[tuple] == 0)continue;
                    byte nonmemberCount = nonmemberCounts[tuple];

                    if(nonmemberCount > 1) {
                        nonmemberCounts[tuple]++;
                    }
                    else if(nonmemberCount == 1){
                        int measureValue = values[tuple];
                        int nonMemberMode = 0;
                        for(int mode = 0; mode < CORE_COLUMN_LENGTH; mode++) {
                            if(mode != maxMode) {
                                if(!slices.get(mode).get(attVals[mode][tuple] - 1).isExists()) {
                                    nonMemberMode = mode;
                                    break;
                                }
                            }
                        }
                        nonmemberCounts[tuple]++;
                        int attVal = attVals[nonMemberMode][tuple];
                        outHeaps[nonMemberMode].updatePriority(attVal, outHeaps[nonMemberMode].getPriority(attVal) - measureValue);
                    }
                    else if(nonmemberCount==0){ //nonmember count == 0;
                        int measureValue = globalValue[tuple];
                        massSumOut += measureValue;
                        nonmemberCounts[tuple]++;
                        for (int mode = 0; mode < globalSize.length - 1; mode++) {
                            int attr = 0;
                            if(mode < Converge.CORE_COLUMN_LENGTH){
                                if(mode != maxMode) {
                                    int attVal = attVals[mode][tuple];
                                    inHeaps[mode].updatePriority(attVal, inHeaps[mode].getPriority(attVal) - measureValue);
                                }
                                attr = attVals[mode][tuple];
                            }else{
                                attr = attVals[mode][tuple];
                            }
                            Map<Integer, Integer> masses = dataOfAttr.get(mode);
                            masses.replace(attr, masses.get(attr) - globalValue[tuple]);
                            Integer attrMass = dataOfAttr.get(mode).get(attr);

                            if(mode >= Converge.CORE_COLUMN_LENGTH){
                                if(attrMass == 0){
                                    slices.get(mode).get(attr - 1).setExists(false);
                                    size[mode]--;
                                }
                            }


                        }
                        values[tuple] = 0;
                    }
                    else {
                        System.out.println("error-2!");
                        System.exit(0);
                    }
                }
                outHeaps[maxMode].insert(attValToRemove, massSumOut);
                slices.get(maxMode).get(attValToRemove - 1).setExists(false);
                size[maxMode]--;
            }
            else { //insert
                Pair<Integer, Integer> pair = outHeaps[maxMode].poll();
                int attValToInsert = pair.getKey();
                //更新mass
                subTensor.setMass(subTensor.getMass() + pair.getValue());

                sumOfCardinalities++;
                attributeToCardinalities[maxMode]++;

                //update degree in
                int massSumIn = 0;
                List<Integer> tuples = globalSlices.get(maxMode).get(attValToInsert - 1).getDataIndexs();
                for (int tuple : tuples) {
                    if(globalValue[tuple] == 0)continue;
                    byte nonmemberCount = nonmemberCounts[tuple];

                    if(nonmemberCount > 2) {
                        nonmemberCounts[tuple]--;
                        continue;
                    }
                    else if(nonmemberCount == 2){
                        int measureValue = globalValue[tuple];
                        int nonMemberMode = 0;
                        for(int mode = 0; mode < CORE_COLUMN_LENGTH; mode++) {
                            if(mode != maxMode) {
                                if(!slices.get(mode).get(attVals[mode][tuple] - 1).isExists()) {
                                    nonMemberMode = mode;
                                    break;
                                }
                            }
                        }
                        nonmemberCounts[tuple]--;
                        int index = attVals[nonMemberMode][tuple];
                        outHeaps[nonMemberMode].updatePriority(index, outHeaps[nonMemberMode].getPriority(index) + measureValue);
                    }
                    else if(nonmemberCount == 1){ //nonmember count == 1;
                        int measurevALUE = globalValue[tuple];
                        massSumIn += measurevALUE;
                        nonmemberCounts[tuple]--;
                        for (int mode = 0; mode < globalSize.length - 1; mode++) {
                            int attr = 0;
                            if(mode < Converge.CORE_COLUMN_LENGTH){
                                if(mode != maxMode) {
                                    int index = attVals[mode][tuple];
                                    inHeaps[mode].updatePriority(index, inHeaps[mode].getPriority(index) + measurevALUE);
                                }
                                attr = attVals[mode][tuple];
                                Map<Integer, Integer> coreMasses = dataOfAttr.get(mode);
                                coreMasses.replace(attr, coreMasses.get(attr) + globalValue[tuple]);
                            }else{
                                attr = attVals[mode][tuple];
                                Map<Integer, Integer> timeMass = dataOfAttr.get(mode);
                                timeMass.replace(attr, timeMass.get(attr) + globalValue[tuple]);
                                if (!slices.get(mode).get(attr - 1).isExists()){
                                    slices.get(mode).get(attr - 1).setExists(true);
                                    size[mode]++;
                                }
                            }
                        }
                        values[tuple] = globalValue[tuple];

                    }
                    else { //nonmemberCount
                        System.out.println(slices.get(maxMode).get(attVals[maxMode][tuple] - 1).isExists());
                        System.out.println("error-1!");
                        System.exit(0);
                    }
                }
                inHeaps[maxMode].insert(attValToInsert, massSumIn);
                slices.get(maxMode).get(attValToInsert - 1).setExists(true);
                size[maxMode]++;
            }

            //更新最小最大经纬度信息
            currentScore = Util.computeDensity(subTensor);

        }
    }

    /**
     * create MinHeaps, which size is the size of coreDimensions.
     * @param subTensor
     * @return
     */
    private static IMinHeap[] createHeaps(Tensor subTensor) {
        List<Map<Integer, Integer>> dataOfAttr = subTensor.getDataOfAttr();
        IMinHeap[] heaps = new IMinHeap[CORE_COLUMN_LENGTH];
        for (int i = 0; i < CORE_COLUMN_LENGTH; i++) {
            IMinHeap heap;
            if(i < TIME_COLUMN){
                heap = new HashIndexedMinHeap(globalSize[i] + 1);
            }else{
                heap = new HashIndexedMinHeap(Converge.globalMaxTime + 1);
            }
            Map<Integer, Integer> massMap = dataOfAttr.get(i);
            for (Integer key : massMap.keySet()) {
                heap.insert(key, massMap.get(key));
            }
            /*for (int j = 1; j < globalSize[i] + 1; j++) {
                int value = massMap.get(j) == null ? 0 : massMap.get(j);
                heap.insert(j,value);
            }*/
            heaps[i]  = heap;
        }
        return heaps;
    }

    /**
     * 针对每一个子张量，通过减少用户或商品维度提升其密度。
     * @param subTensor 时间维度上密集的子张量
     */
    public static void updateCoreDimensionLikeDcube(Tensor subTensor){
        //创建一个用于遍历删除的压缩版subTensor
        Tensor tempTensor = new Tensor(subTensor);
        int mass = tempTensor.getMass();
        int[] cardinality = tempTensor.getSize().clone();
        final int[][] modeToAttVals = createModeToAttVals(cardinality, tempTensor);
        int sumOfCardinality = 0;
        for(int i = 0; i < CORE_COLUMN_LENGTH; i++){
            sumOfCardinality += cardinality[i];
        }
        int[] modeToAliveNum = tempTensor.getSize().clone();
        int[] modeToRemovedNum = new int[CORE_COLUMN_LENGTH];
        List<Map<Integer, Integer>> dataOfAttr = tempTensor.getDataOfAttr();

        int delIndex = 0;
        int[] deleteModes = new int[sumOfCardinality];
        int[] deleteAttrs = new int[sumOfCardinality];

        int maxIters = 0;
        double maxDensityAmongIters = Util.computeDensity2(tempTensor);
        int i = 0;

        while(i < sumOfCardinality){
            int maxMode = 0;
            double maxDensityAmongModes = -Double.MAX_VALUE;
            for(int mode = 0; mode < Converge.CORE_COLUMN_LENGTH; mode++){
                if(modeToAliveNum[mode] > 0){
                    double threshold = mass * 1.0/ modeToAliveNum[mode];
                    int numToRemove = 0;
                    long removedMassSum = 0;
                    Map<Integer, Integer> attValToMass;
                    int[] attVals = modeToAttVals[mode];
                    attValToMass = dataOfAttr.get(mode);

                    for(int j = modeToRemovedNum[mode]; j < cardinality[mode]; j++){
                        int attVal = attVals[j];
                        int attMass = attValToMass.get(attVal) == null ? 0 : attValToMass.get(attVal);
                        if(attMass <= threshold){
                            numToRemove++;
                            removedMassSum += attMass;
                        }
                    }

                     if(numToRemove >= 1){
                        double tempDensity = Util.computeDensity2(mass - removedMassSum, Util.computeVolume2(tempTensor) - numToRemove);
                        if(tempDensity >= maxDensityAmongModes){
                            maxMode = mode;
                            maxDensityAmongModes = tempDensity;
                        }
                    }else{
                        System.out.println("Sanity Check!");
                    }
                }
            }

            double threshold = mass * 1.0 / modeToAliveNum[maxMode];
            final Map<Integer, Integer> attValToMass;
            final boolean[] attValsToRemove = new boolean[cardinality[maxMode]];
            attValToMass = dataOfAttr.get(maxMode);
            //排序过后modeToAttVals的属性值是以属性切片的质量排序的
            sortDeleteSlice(modeToAttVals[maxMode], attValToMass, modeToRemovedNum[maxMode], cardinality[maxMode] - 1);
            int[] attVals = modeToAttVals[maxMode];

            for(int j = modeToRemovedNum[maxMode]; j < cardinality[maxMode]; j++){
                int attVal = attVals[j];
                int attrMass = attValToMass.get(attVal) == null ? 0 : attValToMass.get(attVal);
                if(attrMass <= threshold){
                    mass -= attrMass;
                    double density = Util.computeDensity2(tempTensor.getMass() - attrMass, Util.computeVolume2(tempTensor) - 1);
                    //后续需要更新tempTensor中的Mass和Size
                    tempTensor.setMass(tempTensor.getMass() - attrMass);
                    int[] tempSize = tempTensor.getSize();
                    tempSize[maxMode]--;
                    tempTensor.setSize(tempSize);

                    if(density > maxDensityAmongIters){
                        maxDensityAmongIters = density;
                        maxIters = i + 1;
                    }
                    modeToRemovedNum[maxMode]++;
                    modeToAliveNum[maxMode]--;
                    deleteModes[delIndex] = maxMode;
                    deleteAttrs[delIndex++] = attVal;
                    i++;
                    attValsToRemove[j] = true;
                }else {
                    break;
                }
            }

            removeAndUpdateAttValMasses(tempTensor, maxMode, attValsToRemove, dataOfAttr, attVals);

        }

        //真正的子张量删除阶段
        for(int index = 0; index < maxIters; index++){
            subTensor.deleteAttr(deleteAttrs[index], deleteModes[index]);
        }
    }

    private static void removeAndUpdateAttValMasses(Tensor tensor, int maxMode, boolean[] attValsToRemove, List<Map<Integer, Integer>> masses, int[] attVals) {
        int len = attValsToRemove.length;
        List<Integer> dataList;
        int[] value = tensor.getValue();
        int[] size = tensor.getSize();
        Map<Integer, Integer> massesMap = masses.get(maxMode);
        //1、找到要删除的所有tuple
        //2、根据tuple找到要质量减小的userMass或objectMass
        //3、将tuple的value置为0，将对于删除的userMass或ObjectMass置为0

        for(int i = 0; i < len; i++){
            if(attValsToRemove[i]){
                //将当前维度属性质量置为0
                int attVal = attVals[i];
                massesMap.replace(attVal, 0);
                Slice tempSlice = tensor.getSlices().get(maxMode).get(attVal - 1);
                if(!tempSlice.isExists())continue;
                dataList = tempSlice.getDataIndexs();
                if(dataList == null)continue;
                for(int data : dataList){
                    if(value[data] == 0)continue;
                    for (int j = 0; j < globalSize.length - 1; j++) {
                        if(j == maxMode)continue;
                        int attval = Converge.attVals[j][data];
                        Map<Integer, Integer> tempMap = masses.get(j);
                        int attMass = tempMap.get(attval) == null ? 0 : tempMap.get(attval);
                        tempMap.replace(attval, attMass - value[data]);
                        if(j >= CORE_COLUMN_LENGTH){
                            if(tempMap.get(attVal) == 0){
                                size[j]--;
                            }
                        }
                    }
                    value[data] = 0;
                }
            }
        }
    }

    private static void removeAndUpdateAttValMassesLikeMzoom(Tensor tensor, int maxMode, int attVal, List<Map<Integer, Integer>> masses, IMinHeap[] heaps) {
        List<Integer> dataList;
        int[] value = tensor.getValue();
        int[] size = tensor.getSize();
        Map<Integer, Integer> massesMap = masses.get(maxMode);
        //1、找到要删除的所有tuple
        //2、根据tuple找到要质量减小的userMass或objectMass
        //3、将tuple的value置为0，将对于删除的userMass或ObjectMass置为0

        //将当前维度属性质量置为0
        massesMap.replace(attVal, 0);
        Slice tempSlice = tensor.getSlices().get(maxMode).get(attVal - 1);
        dataList = tempSlice.getDataIndexs();
        for(int data : dataList){
            if(value[data] == 0)continue;
            for (int j = 0; j < globalSize.length - 1; j++) {
                if(j == maxMode)continue;
                int att = attVals[j][data];
                Map<Integer, Integer> tempMap = masses.get(j);
                if(j < CORE_COLUMN_LENGTH){
                    heaps[j].updatePriority(att, tempMap.get(att) - value[data]);
                }
                tempMap.replace(att, tempMap.get(att) - value[data]);
                if(j >= CORE_COLUMN_LENGTH){
                    if(tempMap.get(att) == 0){
                        size[j]--;
                    }
                }
            }
            value[data] = 0;
        }
    }

    /**
     * 将attributes数组中的属性值，按照其切片质量排序
     * @param attributes : 属性值列表
     * @param masses ：输出某维度中某属性的数据量
     */
    private static void sortDeleteSlice(int[] attributes, Map<Integer, Integer> masses, int left, int right) {

        if (attributes == null || attributes.length == 0)
            return;

        if (left >= right)
            return;

        int middle = left + (right - left) / 2;
        int pivot = masses.get(attributes[middle]) == null ? 0 : masses.get(attributes[middle]);

        int i = left, j = right;
        while (i <= j) {
            while ((masses.get(attributes[i]) == null ? 0 : masses.get(attributes[i])) < pivot) {
                i++;
            }

            while ((masses.get(attributes[j]) == null ? 0 : masses.get(attributes[j])) > pivot) {
                j--;
            }

            if (i <= j) {
                int temp = attributes[i];
                attributes[i] = attributes[j];
                attributes[j] = temp;
                i++;
                j--;
            }
        }

        if (left < j)
            sortDeleteSlice(attributes, masses, left, j);

        if (right > i)
            sortDeleteSlice(attributes, masses, i, right);
    }

    /**
     * 创建一个二维数组，可以根据其维度和索引锁定一个属性
     * 后续会使下标索引变成按质量排序的，从0-len为从质量小到大的属性切片
     * @return ：返回从1开始的属性值列表
     */
    private static int[][] createModeToAttVals(int[] cardinality) {
        int[][] modeToIndexs = new int[CORE_COLUMN_LENGTH][];
        for(int mode = 0; mode < CORE_COLUMN_LENGTH; mode++){
            int[] indices = new int[cardinality[mode]];
            for(int index = 0; index < cardinality[mode]; index++){
                indices[index] = index + 1;
            }
            modeToIndexs[mode] = indices;
        }
        return modeToIndexs;
    }

    /**
     * 创建一个二维数组，按照顺序存放质量不为0的属性索引
     * @return
     */
    private static int[][] createModeToAttVals(int[] cardinality, Tensor subTensor) {
        int[][] modeToIndices = new int[CORE_COLUMN_LENGTH][];
        // int[] userMass = subTensor.getUserMass();
        // int[] objectMass = subTensor.getObjectMass();
        for(int mode = 0; mode < CORE_COLUMN_LENGTH; mode++){
            Map<Integer, Integer> masses = subTensor.getDataOfAttr().get(mode);
            int[] indices = new int[cardinality[mode]];
            int num = 0;
            for (Integer key : masses.keySet()) {
                if(masses.get(key) != 0){
                    indices[num++] = key;
                }
            }
            modeToIndices[mode] = indices;
        }
        return modeToIndices;
    }

    /**
     * 对tensor中的切片进行整理
     *
     * @return ：返回按照质量排好序的属性值列表，其中0为维度，1为属性值，2为该属性的质量
     */
    private static int[][] orderSliceByDensity(Tensor tensor) {
        int size = tensor.getSize()[TIME_COLUMN] + tensor.getSize()[LATITUDE_COLUMN] + tensor.getSize()[LONGITUDE_COLUMN];
        int[][] sliceInformation = new int[3][size];
        List<List<Slice>> slices = tensor.getSlices();
        int[] value = tensor.getValue();

        // 计算出每一个切片的质量，按照质量对切片进行排序
        // 将time、longitude、latitude维度的切片统一到一个列表中，加上维度信息
        int index = 0;
        int existSize = 0;
        for (int i = TIME_COLUMN; i < globalSize.length - 1; i++) {
            List<Slice> sliceList = slices.get(i);
            for (Slice s : sliceList) {
                if (s.isExists()) {
                    existSize++;
                    sliceInformation[0][index] = s.getIndex();
                    sliceInformation[1][index] = i;

                    // 计算切片质量
                    int tempMass = 0;
                    Vector<Integer> dataIndexs = s.getDataIndexs();
                    for (Integer data : dataIndexs) {
                        if (value[data] != 0) tempMass += value[data];
                    }
                    sliceInformation[2][index++] = tempMass;
                }
            }
        }

        sort(sliceInformation, 0, existSize - 1);

        return sliceInformation;
    }

    /**
     * 将attributes数组中的属性值和维度，按照其切片质量排序
     *
     * @param sliceInformation：其中第0行为切片质量，第2行为切片维度，第三行为切片质量
     */
    private static void sort(int[][] sliceInformation, int left, int right) {

        if (sliceInformation == null || sliceInformation.length == 0) return;

        if (left > right) return;

        int pivot = sliceInformation[2][left];
        int i = left, j = right;

        while (i != j) {
            while (sliceInformation[2][j] >= pivot && i < j) {
                j--;
            }

            while (sliceInformation[2][i] <= pivot && i < j) {
                i++;
            }

            int temp1 = sliceInformation[0][i];
            sliceInformation[0][i] = sliceInformation[0][j];
            sliceInformation[0][j] = temp1;
            int temp2 = sliceInformation[1][i];
            sliceInformation[1][i] = sliceInformation[1][j];
            sliceInformation[1][j] = temp2;
            int temp3 = sliceInformation[2][i];
            sliceInformation[2][i] = sliceInformation[2][j];
            sliceInformation[2][j] = temp3;
        }

        if(i == j){
            sliceInformation[0][left] = sliceInformation[0][i];
            sliceInformation[0][i] = sliceInformation[0][left];
            sliceInformation[1][left] = sliceInformation[1][i];
            sliceInformation[1][i] = sliceInformation[1][left];
            sliceInformation[2][left] = sliceInformation[2][i];
            sliceInformation[2][i] = pivot;
            sort(sliceInformation, left, i - 1);
            sort(sliceInformation, j + 1, right);
        }

    }

    /**
     * 根据输入数据，将数据转换为张量进行存储
     * @param file_path：文件输入路径
     * @param splitStr：文件数据分隔符
     * @return 原始数据张量
     */
    public static Tensor createTensor(String file_path, String splitStr){
        String line;
        String[] attributes;
        int[] size = Arrays.copyOfRange(globalSize, 0, globalSize.length - 1);
        Arrays.fill(size, TIME_COLUMN, size.length, 0);
        attVals = new int[globalSize.length - 1][globalSize[globalSize.length - 1]];
        int[] value = new int[globalSize[globalSize.length - 1]];
        int mass = 0;
        List<Map<Integer, Integer>> dataOfAttr = new ArrayList<>();
        //初始化dataOfAttr
        for (int i = 0; i < globalSize.length - 1; i++) {
            dataOfAttr.add(new HashMap<>());
        }

        BufferedReader reader;
        int i = 0; //显示当前记录为第几条记录

        //modeId用于给属性值进行转换,初始化modeId,全部填充为1
        Integer[] modeId = new Integer[globalSize.length - 3];
        Arrays.fill(modeId,1);

        //初始化slices
        List<List<Slice>> slices = new ArrayList<>();
        //初始化user、item、time维度slices
        for (int j = 0; j < globalSize.length - 4; j++) {
            List<Slice> tempSlices = new ArrayList<>();
            for (int k = 0; k < globalSize[j]; k++) {
                Slice tempSlice = new Slice(k + 1,  true);
                tempSlices.add(tempSlice);
            }
            slices.add(tempSlices);
        }

        //初始化时间、经纬度维度的slices
        List<Slice> tempSlices1 = new ArrayList<>();
        for (int k = 0; k < globalMaxTime; k++) {
            tempSlices1.add(new Slice(k + 1,  false));
        }
        slices.add(tempSlices1);
        List<Slice> tempSlices2 = new ArrayList<>();
        for (int k = 0; k < globalMaxLatitude; k++) {
            tempSlices2.add(new Slice(k + 1,  false));
        }
        slices.add(tempSlices2);
        List<Slice> tempSlices3 = new ArrayList<>();
        for (int k = 0; k < globalMaxlongitude; k++) {
            tempSlices3.add(new Slice(k + 1,  false));
        }
        slices.add(tempSlices3);

        for (int j = 0; j < globalSize.length - 3; j++) {
            convert.add(new HashMap<>());
            recoverDicts.add(new HashMap<>());
        }

        try {
            reader = new BufferedReader(new FileReader(file_path));
            while ((line = reader.readLine()) != null){
                attributes = line.split(splitStr);
                value[i] = Integer.parseInt(attributes[globalSize.length - 1]);
                for(int index = 0; index < globalSize.length - 4; index++){
                    //转换数据格式并存储属性映射关系,导入数据到attVals、value,存储每一个维度的质量
                    Map<String ,Integer> currentMap = convert.get(index);
                    Map<Integer, String> recoverMap = recoverDicts.get(index);
                    String key = attributes[index];
                    if(!currentMap.containsKey(key)){
                        currentMap.put(key, modeId[index]);
                        recoverMap.put(modeId[index]++, key);
                    }
                    attVals[index][i] = currentMap.get(key);
                    //修改dataOfAttr
                    Map<Integer,Integer> datas = dataOfAttr.get(index);
                    datas.putIfAbsent(attVals[index][i], 0);
                    Integer tempValue = datas.get(attVals[index][i]);
                    datas.replace(attVals[index][i], tempValue + value[i]);
                }

                //处理时间维度
                attVals[TIME_COLUMN][i] = Integer.parseInt(attributes[TIME_COLUMN]);
                Map<Integer, Integer> datas3 = dataOfAttr.get(TIME_COLUMN);
                datas3.putIfAbsent(attVals[TIME_COLUMN][i], 0);
                Integer tempValue = datas3.get(attVals[TIME_COLUMN][i]);
                datas3.replace(attVals[TIME_COLUMN][i], tempValue + value[i]);
                if(!slices.get(TIME_COLUMN).get(attVals[TIME_COLUMN][i] - 1).isExists()){
                    slices.get(TIME_COLUMN).get(attVals[TIME_COLUMN][i] - 1).setExists(true);
                    size[TIME_COLUMN]++;
                }

                //统计总质量
                mass += value[i];

                //处理地址的经纬度
                int latitude = Integer.parseInt(attributes[LATITUDE_COLUMN]);
                int longitude = Integer.parseInt(attributes[LONGITUDE_COLUMN]);
                attVals[LATITUDE_COLUMN][i] = latitude;
                attVals[LONGITUDE_COLUMN][i] = longitude;
                //修改经纬度的dataOfAttr
                Map<Integer, Integer> datas1 = dataOfAttr.get(LATITUDE_COLUMN);
                Map<Integer, Integer> datas2 = dataOfAttr.get(LONGITUDE_COLUMN);
                //使经度对应的keyValue自增1
                datas1.putIfAbsent(latitude, 0);
                Integer tempLatitudeValue = datas1.get(latitude);
                datas1.replace(latitude, tempLatitudeValue + value[i]);
                if(!slices.get(LATITUDE_COLUMN).get(latitude - 1).isExists()){
                    slices.get(LATITUDE_COLUMN).get(latitude - 1).setExists(true);
                    size[LATITUDE_COLUMN]++;
                }
                //使维度对应的keyValue自增1
                datas2.putIfAbsent(longitude, 0);
                Integer tempLongitudeValue = datas2.get(longitude);
                datas2.replace(longitude, tempLongitudeValue + value[i]);
                if(!slices.get(LONGITUDE_COLUMN).get(longitude - 1).isExists()){
                    slices.get(LONGITUDE_COLUMN).get(longitude - 1).setExists(true);
                    size[LONGITUDE_COLUMN]++;
                }

                //更新user、item、time纬度的slices信息
                for (int mode = 0; mode < globalSize.length - 1; mode++) {
                    int attr = 0;
                    if(mode < globalSize.length - 4){
                        Map<String ,Integer> currentMap = convert.get(mode);
                        attr = currentMap.get(attributes[mode]);
                    } else if (mode == TIME_COLUMN) {
                        attr = Integer.parseInt(attributes[TIME_COLUMN]);
                    } else if(mode == LATITUDE_COLUMN){
                        attr = latitude;
                    }else if(mode == LONGITUDE_COLUMN){
                        attr = longitude;
                    }
                    //取出当前维度的slices
                    List<Slice> tempSlices = slices.get(mode);
                    Slice slice = tempSlices.get(attr - 1);
                    //修改切片包含的数据索引
                    slice.dataIndexs.add(i);
                }

                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new Tensor(globalMinTime, globalMaxTime, globalMinLatitude, globalMaxLatitude, globalMinlongitude, globalMaxlongitude, mass , size, value, slices, dataOfAttr);
    }

    /**
     * 根据子张量数组和输出文件路径，将子张量输出到磁盘中
     *
     * @param tensorList  子张量数组
     * @param output_path 输出路径
     */
    public static void outputTensorList(List<Tensor> tensorList, String output_path, int totalNum) {
        File directory;
        try {
            //1、先判断文件加是否存在？不存在则创建对应文件夹
            directory = new File(output_path);
            if(!directory.exists() || !directory.isDirectory()){
                //文件夹不存在或不是文件夹，新建对应文件夹
                directory.mkdirs();
            }
            //2、遍历子张量数组，每一个张量存在一个文件中
            for (int i = 0; i < totalNum; i++) {
                Tensor subTensor = tensorList.get(i);
                double timeRange = Util.computeTemporalRange(subTensor.getMinTime(), subTensor.getMaxTime()) * 100;
                double spaceRange = Util.computeSpatialRange(subTensor.getMinLatitude(), subTensor.getMaxLatitude(), subTensor.getMinLongitude(), subTensor.getMaxLongitude()) * 100;
                int[] size = subTensor.getSize();
                System.out.println("Tensor " + (i + 1) + ":");
                System.out.println("Mass: " + subTensor.getMass());
                System.out.println("Valume: " + Util.computeVolume(subTensor));
                System.out.println("Density: " + Util.computeDensity(subTensor));
                // System.out.println("score: " + Util.computeDensity(subTensor));
                StringBuilder s = new StringBuilder();
                for (int j = 0; j < globalSize.length - 4; j++) {
                    s.append(size[j]).append(" x ");
                }
                s.replace(s.length() - 3, s.length(), "");
                System.out.println("The size of the " + (i + 1) + "th sub-tensor is " + s);

                subTensor.updatePeakInfomation();

                //计算时间范围并输出
                System.out.println("该子张量时间跨度为：" + timeRange + "%");
                System.out.println("该子张量的空间范围为：" + spaceRange + "%");

                //将指定子张量属性写到磁盘中
                outputProperties(subTensor, i + 1, output_path);

                //将指定子张量数据写到磁盘中
                ourputData(subTensor, i + 1, output_path);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 输出对应子张量的各维度的属性和整体信息
     * @param subTensor : 需要输出的子张量
     * @param i ： 第几个密集子张量
     * @param output_path ： 输出路径
     */
    public static void outputProperties(Tensor subTensor, int i, String output_path) throws IOException {
        String subTensorProFilePath;
        File subTensorProFile;
        FileWriter wt1;
        int[] size = subTensor.getSize();
        subTensorProFilePath = output_path + "/" + "block_" + i + ".properties";
        subTensorProFile = new File(subTensorProFilePath);

        //输出张量属性文件
        if(subTensorProFile.exists()){
            subTensorProFile.delete();
        }else{
            subTensorProFile.createNewFile();
        }
        wt1 = new FileWriter(subTensorProFilePath);

        //找出每一个维度存在的属性
        List<List<Integer>> modeAttrs = new ArrayList<>();
        List<List<Slice>> slices = subTensor.getSlices();
        for(int mode = 0; mode < CORE_COLUMN_LENGTH; mode++){
            List<Slice> sliceList = slices.get(mode);
            List<Integer> attrs = new ArrayList<>();
            for(Slice slice : sliceList){
                if(slice.isExists()){
                    attrs.add(slice.getIndex());
                }
            }
            modeAttrs.add(attrs);
        }

        //希望的文件格式：
        StringBuilder sb1 = new StringBuilder();
        sb1.append("Tensor " + i + ":\n");
        sb1.append("Mass: "+ subTensor.getMass() +"\n");
        sb1.append("Valume: "+ Util.computeVolume(subTensor) +"\n");
        sb1.append("Density: "+ Util.computeDensity(subTensor.getMass(), Util.computeVolume(subTensor)) +"\n");
        sb1.append("The size of the " + i + "th sub-tensor is ");
        for (int j = 0; j < CORE_COLUMN_LENGTH; j++) {
            sb1.append(size[j] + " x ");
        }
        for (int j = 0; j < CORE_COLUMN_LENGTH; j++) {
            sb1.append(j + ":\n");
            Map<Integer, String> recoverDict = recoverDicts.get(j);
            List<Integer> modeAttr = modeAttrs.get(j);
            for (Integer attr : modeAttr) {
                sb1.append(recoverDict.get(attr) + ", ");
            }
            sb1.replace(sb1.length() - 2, sb1.length(), "\n");
        }

        System.out.println();
        wt1.append(sb1);
        wt1.flush();
        wt1.close();
    }

    public static void ourputData(Tensor subTensor, int i, String output_path) throws IOException {
        String subTensorDataFilePath;
        File subTensorDataFile;
        FileWriter wt2;
        subTensorDataFilePath = output_path + "/" + "block_" + i + ".tuples";
        subTensorDataFile = new File(subTensorDataFilePath);

        //输出张量数据文件
        if(subTensorDataFile.exists()){
            subTensorDataFile.delete();
        }else{
            subTensorDataFile.createNewFile();
        }
        wt2 = new FileWriter(subTensorDataFile);
        StringBuilder sb2 = new StringBuilder();
        //获取恢复数据的字典
        Map<Integer, String> timeRecoverDict = recoverDicts.get(TIME_COLUMN);
        int[] values = subTensor.getValue();
        //遍历子张量selectDataByTime，获得记录位置信息
        //希望输出的格式信息：
        //通过记录位置信息结合attVals、value输出该记录
        //收集subTensor的data
        for (int k = 0; k < values.length; k++) {
            if(values[k] == 0)continue;
            for(int mode = 0; mode < globalSize.length - 4; mode++ ){
                sb2.append(recoverDicts.get(mode).get(attVals[mode][k])).append(",");
            }
            sb2.append(attVals[TIME_COLUMN][k] + "," + attVals[Converge.LATITUDE_COLUMN][k] + "," + attVals[Converge.LONGITUDE_COLUMN][k] + "," + values[k] + "\n");
        }
        wt2.append(sb2);
        wt2.flush();
        wt2.close();
    }

}
