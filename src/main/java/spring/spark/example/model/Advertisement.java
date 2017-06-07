package spring.spark.example.model;

/**
 * ������  �� ���� 
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
	 * ����
	 */
	private float  order;
	/**
	 * ��(ī�װ� ��ȣ)
	 */
	private double label;
	
	/**
	 * Ȯ�� 
	 */
	private double probability;
	
	/**
	 * Ű����
	 */
	private String words;
	
	/**
	 * �������� ī�װ� ����
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
