namespace wtpsplit;

/// <summary>Defines how overlapping logits are weighted during aggregation.</summary>
public enum WeightingType {
    /// <summary>Equal weight for all overlapping positions.</summary>
    Uniform,
    /// <summary>Apply hat-shaped (triangular) weighting favoring center positions.</summary>
    Hat
}
