import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Util {

    public static double sigmoid_base = 0;

    public static void setSigmoid_base(double sigmoid_base) {
        Util.sigmoid_base = sigmoid_base;
    }

    /**
     * 统计每一个维度的大小
     * @param file_path：输入文件路径
     * @param splitStr：数据文件分隔符
     * @return 三个维度的大小和记录的条数
     */
    public static int[] computerSize(String file_path, String splitStr) throws Exception {
        BufferedReader reader = null;
        int dimensions = 0;
        //提前读取一行，判断其维度
        try {
            reader = new BufferedReader(new FileReader(file_path));
            dimensions = reader.readLine().split(",").length;
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if(reader != null){
                reader.close();
            }
        }

        //统计三个维度的大小和记录的条数
        int[] size = new int[dimensions];
        String[] attributes;
        List<Set<String>> dimensionList = new ArrayList<>();
        for(int i = 0; i < dimensions - 1; i++){
            dimensionList.add(new HashSet<>());
        }
        String line;
        try {
            reader = new BufferedReader(new FileReader(file_path));
            while((line = reader.readLine()) != null){
                //统计记录条数
                size[dimensions - 1]++;
                attributes = line.split(splitStr);
                for(int i = 0; i < dimensions - 1; i++){
                    dimensionList.get(i).add(attributes[i]);
                    //统计最大最小时间
                    Converge.globalMinTime = Math.min(Integer.parseInt(attributes[dimensions - 4]), Converge.globalMinTime);
                    Converge.globalMaxTime = Math.max(Integer.parseInt(attributes[dimensions - 4]), Converge.globalMaxTime);
                    Converge.globalMinLatitude = Math.min(Integer.parseInt(attributes[dimensions - 3]), Converge.globalMinLatitude);
                    Converge.globalMaxLatitude = Math.max(Integer.parseInt(attributes[dimensions - 3]), Converge.globalMaxLatitude);
                    Converge.globalMinlongitude = Math.min(Integer.parseInt(attributes[dimensions - 2]), Converge.globalMinlongitude);
                    Converge.globalMaxlongitude = Math.max(Integer.parseInt(attributes[dimensions - 2]), Converge.globalMaxlongitude);
                }
            }
            for(int i = 0; i < dimensions - 1; i++){
                size[i] = dimensionList.get(i).size();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            try {
                if(reader != null){
                    reader.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return size;

    }

    /**
     * 转换范围为-90-90的纬度变为0-180的纬度范围
     * @param num
     * @return
     */
    public static int ceilSpaceByLatitude(double num){
        if(num == -90)return 1;
        return (int) Math.ceil(num + 90);
    }

    /**
     * 转换范围为-180-180的经度到0-360的经度范围
     * @param num
     * @return
     */
    public static int ceilSpaceBylongitude(double num){
        if(num == -180)return 1;
        return (int) Math.ceil(num + 180);
    }

    /**
     * 计算子张量密度
     *
     * @return
     */
    public static double computeDensity(long mass, long size){
        return size == 0 ? 0 : mass * 1.0 / size * (Converge.globalSize.length - 1);
    }

    public static double computeDensity2(long mass, long size){
        return size == 0 ? 0 : mass * 1.0 / size * (Converge.CORE_COLUMN_LENGTH);
    }

    public static double computeDensity(Tensor tensor){
        // return tensor.getMass() * (float)Converge.CORE_COLUMN_LENGTH / computeVolume(tensor);
        return tensor.getMass() * ((float)Converge.globalSize.length - 1) / computeVolume(tensor);
    }

    public static double computeDensity2(Tensor tensor){
        return tensor.getMass() * (float)Converge.CORE_COLUMN_LENGTH / computeVolume2(tensor);
        // return tensor.getMass() * ((float)Converge.globalSize.length - 1) / computeVolume(tensor);
    }

    public static boolean isDense(Tensor tensor){
        return tensor.getMass() * (float)Converge.CORE_COLUMN_LENGTH / computeVolume(tensor) > 1;
    }

/*    public static double computeWeightDensity(Tensor tensor){
        int[] size = tensor.getSize();
        double coreSize = 0;
        for(int i = 0; i < Converge.globalSize.length - 4; i++){
            coreSize += size[i];
        }
        double core = coreSize * Converge.CORE_PROPORTION;
        double time = (tensor.getMaxTime() - tensor.getMinTime() + 1) * Converge.timeStep * Converge.TIME_PROPORTION;
        double space = (((tensor.getMaxLatitude() - tensor.getMinLatitude()) * Converge.latitudeStep * 0.5)
                + ((tensor.getMaxLongitude() - tensor.getMinLongitude()) * Converge.longitudeStep * 0.5)) * Converge.SPACE_PROPORTION;
        return tensor.getMass() * 1.0 / (core + time + space);
    }*/

/*    public static double computeDenominator(List<Set<Integer>> totalSize, int minTime, int maxTime, int minLatitude, int maxLatitude, int minLongitude, int maxLongitude){
        double coreSize = 0;
        for(int i = 0; i < Converge.globalSize.length - 4; i++){
            coreSize += totalSize.get(i).size();
        }
        double core = coreSize * Converge.CORE_PROPORTION;
        double time = (maxTime - minTime + 1) * Converge.timeStep * Converge.TIME_PROPORTION;
        double space = (((maxLatitude - minLatitude) * Converge.latitudeStep * 0.5)
                + ((maxLongitude - minLongitude) * Converge.longitudeStep * 0.5)) * Converge.SPACE_PROPORTION;
        return core + time + space;
    }*/

    /**
     * 计算子张量的体积，也可称为size
     * @param tensor 子张量
     * @return 子张量体积
     */
    public static long computeVolume(Tensor tensor){
        int[] size = tensor.getSize();
        int coreSize = 0;
        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            coreSize += size[i];
        }
        return coreSize;
    }

    public static long computeVolume2(Tensor tensor){
        int[] size = tensor.getSize();
        int coreSize = 0;
        for(int i = 0; i < Converge.CORE_COLUMN_LENGTH; i++){
            coreSize += size[i];
        }
        return coreSize;
    }

    public static double computeTSDensity(Tensor tensor, int mode){
        return tensor.getMass() * 1.0 / tensor.getSize()[mode];
    }

    public static double computeTimeDensity(Tensor tensor){
        return tensor.getMass() * 1.0 / tensor.getSize()[Converge.TIME_COLUMN];
    }

    public static double computeSpaceDensity(Tensor tensor){
        return tensor.getMass() * 1.0 / (tensor.getSize()[Converge.LATITUDE_COLUMN] + tensor.getSize()[Converge.LONGITUDE_COLUMN]);
    }

    public static double computeTSDensity(int mass, int size){
        return mass * 1.0 / size;
    }

    public static double computeTimeDensity(int mass, int size){
        return mass * 1.0 / size;
    }

    public static double computeSpaceDensity(int mass, int size){
        return mass * 2.0 / size;
    }

    private static double computeTSVolume(Tensor tensor) {
        int[] size = tensor.getSize();
        double res = 0;
        for(int i = Converge.TIME_COLUMN; i <= Converge.LONGITUDE_COLUMN; i++){
            res += size[i];
        }
        return res;
    }


    public static double computeTemporalRange(int minTime, int maxTime) {
        return (maxTime - minTime + 1) * 1.0 / (Converge.globalMaxTime - Converge.globalMinTime + 1);
    }

    public static double computeTemporalRange(Tensor tensor) {
        int minTime = tensor.getMinTime();
        int maxTime = tensor.getMaxTime();
        return computeTemporalRange(minTime, maxTime);
    }

    public static double computeTemporalRangeReciprocal(Tensor tensor) {
        int minTime = tensor.getMinTime();
        int maxTime = tensor.getMaxTime();
        return 1 / computeTemporalRange(minTime, maxTime);
    }

    public static double computeSpatialRange(double minLatitude, double maxLatitude, double minLongitude, double maxLongitude) {
        double localArea = (maxLatitude - minLatitude + 1)
                * (maxLongitude - minLongitude + 1);
        double globalArea = (Converge.globalMaxLatitude - Converge.globalMinLatitude + 1)
                * (Converge.globalMaxlongitude - Converge.globalMinlongitude + 1);
        return localArea / globalArea;
    }

    public static double computeMeasurement(double density, double spatial){
        double sigmoid = 1.0 / (1 + Math.exp(-density + sigmoid_base));
        return sigmoid * Converge.DENSITY_PROPORTION + (1 - spatial) * Converge.SPATIAL_PROPORTION;
    }

    public static double computeMeasurementByList(List<Integer> globalList) {
        int mass = 0;
        int totalSize = 0;
        int minTime = Integer.MAX_VALUE;
        int maxTime = Integer.MIN_VALUE;
        int minLatitude = Integer.MAX_VALUE;
        int maxLatitude = Integer.MIN_VALUE;
        int minLongitude = Integer.MAX_VALUE;
        int maxLongitude = Integer.MIN_VALUE;
        int[] value = Converge.tensor.getValue();
        List<Set<Integer>> size = new ArrayList<>();
        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            size.add(new HashSet<>());
        }

        for (Integer index : globalList) {
            mass += value[index];
            int time = Converge.attVals[Converge.TIME_COLUMN][index];
            int latitude = Converge.attVals[Converge.LATITUDE_COLUMN][index];
            int longitude = Converge.attVals[Converge.LONGITUDE_COLUMN][index];
            minTime = Math.min(time, minTime);
            maxTime = Math.max(time, maxTime);
            minLatitude = Math.min(latitude, minLatitude);
            maxLatitude = Math.max(latitude, maxLatitude);
            minLongitude = Math.min(longitude, minLongitude);
            maxLongitude = Math.max(longitude, maxLongitude);
            for(int i = 0; i < Converge.globalSize.length - 1; i++){
                size.get(i).add(Converge.attVals[i][index]);
            }
        }

        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            totalSize += size.get(i).size();
        }

        double density = mass * 1.0 / totalSize * (Converge.globalSize.length - 1);
        setSigmoid_base(density);
        double timeRangeBefore = computeTemporalRange(minTime, maxTime);
        double spatialRangeBefore = computeSpatialRange(minLatitude, maxLatitude, minLongitude, maxLongitude);

        return computeMeasurement(density, spatialRangeBefore);
        // return density;

    }

    public static double computeMeasurementByListNotSet(List<Integer> globalList) {
        int mass = 0;
        int totalSize = 0;
        int minTime = Integer.MAX_VALUE;
        int maxTime = Integer.MIN_VALUE;
        int minLatitude = Integer.MAX_VALUE;
        int maxLatitude = Integer.MIN_VALUE;
        int minLongitude = Integer.MAX_VALUE;
        int maxLongitude = Integer.MIN_VALUE;
        int[] value = Converge.tensor.getValue();
        List<Set<Integer>> size = new ArrayList<>();
        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            size.add(new HashSet<>());
        }

        for (Integer index : globalList) {
            mass += value[index];
            int time = Converge.attVals[Converge.TIME_COLUMN][index];
            int latitude = Converge.attVals[Converge.LATITUDE_COLUMN][index];
            int longitude = Converge.attVals[Converge.LONGITUDE_COLUMN][index];
            minTime = Math.min(time, minTime);
            maxTime = Math.max(time, maxTime);
            minLatitude = Math.min(latitude, minLatitude);
            maxLatitude = Math.max(latitude, maxLatitude);
            minLongitude = Math.min(longitude, minLongitude);
            maxLongitude = Math.max(longitude, maxLongitude);
            for(int i = 0; i < Converge.globalSize.length - 1; i++){
                size.get(i).add(Converge.attVals[i][index]);
            }
        }

        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            totalSize += size.get(i).size();
        }

        double density = mass * 1.0 / totalSize * (Converge.globalSize.length - 1);
        double timeRangeBefore = computeTemporalRange(minTime, maxTime);
        double spatialRangeBefore = computeSpatialRange(minLatitude, maxLatitude, minLongitude, maxLongitude);

        return computeMeasurement(density, spatialRangeBefore);
        // return density;

    }

    public static double preDeleteSlice(int attr, int mode, List<Integer> globalList) {

        int mass = 0;
        int totalSize = 0;
        int minTime = Integer.MAX_VALUE;
        int maxTime = Integer.MIN_VALUE;
        int minLatitude = Integer.MAX_VALUE;
        int maxLatitude = Integer.MIN_VALUE;
        int minLongitude = Integer.MAX_VALUE;
        int maxLongitude = Integer.MIN_VALUE;
        int[] value = Converge.tensor.getValue();
        List<Set<Integer>> size = new ArrayList<>();
        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            size.add(new HashSet<>());
        }

        for (Integer index : globalList) {
            if(Converge.attVals[mode][index] == attr)continue;
            mass += value[index];
            int time = Converge.attVals[Converge.TIME_COLUMN][index];
            int latitude = Converge.attVals[Converge.LATITUDE_COLUMN][index];
            int longitude = Converge.attVals[Converge.LONGITUDE_COLUMN][index];
            minTime = Math.min(time, minTime);
            maxTime = Math.max(time, maxTime);
            minLatitude = Math.min(latitude, minLatitude);
            maxLatitude = Math.max(latitude, maxLatitude);
            minLongitude = Math.min(longitude, minLongitude);
            maxLongitude = Math.max(longitude, maxLongitude);
            for(int i = 0; i < Converge.globalSize.length - 1; i++){
                size.get(i).add(Converge.attVals[i][index]);
            }
        }

        for(int i = 0; i < Converge.globalSize.length - 1; i++){
            totalSize += size.get(i).size();
        }

        double density = mass * 1.0 / totalSize * (Converge.globalSize.length - 1);
        setSigmoid_base(density);
        double timeRangeBefore = computeTemporalRange(minTime, maxTime);
        double spatialRangeBefore = computeSpatialRange(minLatitude, maxLatitude, minLongitude, maxLongitude);

        return computeMeasurement(density, spatialRangeBefore);

    }

    public static void deleteSlice(int attr, int mode, List<Integer> globalList) {
        List<Integer> removeList = new ArrayList<>();
        for (Integer index : globalList) {
            if(Converge.attVals[mode][index] == attr)removeList.add(index);
        }
        globalList.removeAll(removeList);
    }

    public static List<Integer> deleteSlices(Set<Integer> deleteSlices, int mode, List<Integer> globalList) {
        List<Integer> copy = globalList;
        globalList = new ArrayList<>();
        for(Integer index : copy){
            Integer attr = Converge.attVals[mode][index];
            if(!deleteSlices.contains(attr)){
                globalList.add(index);
            }
        }
        return globalList;
    }
}
