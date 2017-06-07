package spring.spark.example.model;

/**
 * 광고예측  모델 순위 
 * @author skanprobability
 *
 */
public class Advertisement {
	
	public Advertisement(float order, double label, double probability, String words) {
		super();
		this.order = order;
		this.label = label;
		this.probability = probability;
		this.words = words;
	}
	
	public Advertisement(float order, double label, double probability, String words , String originalCategory) {
		super();
		this.order = order;
		this.label = label;
		this.probability = probability;
		this.words = words;
		this.originalCategory = originalCategory; 
	}
	/**
	 * 순서
	 */
	private float  order;
	/**
	 * 라벨(카테고리 번호)
	 */
	private double label;
	
	/**
	 * 확율 
	 */
	private double probability;
	
	/**
	 * 키워드
	 */
	private String words;
	
	/**
	 * 오리지널 카테고리 정보
	 */
	private String originalCategory;
	
	public String getOriginalCategory() {
		return originalCategory;
	}
	public void setOriginalCategory(String originalCategory) {
		this.originalCategory = originalCategory;
	}
	public float getOrder() {
		return order;
	}
	public void setOrder(float order) {
		this.order = order;
	}
	public double getLabel() {
		return label;
	}
	public void setLabel(double label) {
		this.label = label;
	}
	public double getProbability() {
		return probability;
	}
	public void setProbability(double probability) {
		this.probability = probability;
	}
	public String getWords() {
		return words;
	}
	public void setWords(String words) {
		this.words = words;
	}
	@Override
	public String toString() {
		return "Advertisement [order=" + order + ", label=" + label + ", probability=" + probability + ", words="
				+ words + "]";
	}
	
}
