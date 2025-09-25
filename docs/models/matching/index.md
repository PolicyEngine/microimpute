# Hot-Deck Matching

The `Matching` model implements imputation through an elegant nearest neighbor distance hot deck matching approach. This technique draws from the principles of statistical matching, using existing complete records (donors) to provide values for records with missing data (recipients) by establishing meaningful connections based on similarities in predictor variables.

## Variable type support

The matching model can handle any variable type—numerical, categorical, boolean, or mixed. Since it transfers actual observed values from similar records rather than generating predictions, it naturally preserves the original data type and distribution of each variable.

## How it works

Statistical or hot-deck matching in Microimpute builds upon the foundation of R's StatMatch package, accessed through the rpy2 interface to provide a seamless integration of R's statistical power with Python's flexibility. The implementation leverages the well-established nearest neighbor distance hot deck matching algorithm, which has a strong theoretical foundation in statistical literature.

During the fitting phase, the model carefully preserves both the complete donor dataset and the relevant variable names that will guide the matching process. This stored information becomes the knowledge base from which the model will draw when making imputations.

The prediction stage initiates a deliberate matching process where each record in the test dataset (the recipients) is systematically compared with the stored donor records. The comparison calculates similarity distances based on the predictor variables, identifying the donor records that most closely resemble each recipient. The matching algorithm efficiently navigates the multidimensional space defined by the predictor variables to find optimal donor-recipient pairs.

Once the matching is complete, the model transfers the values from the matched donors to the recipients for the specified imputed variable. This transfer preserves the natural relationships and patterns present in the original data, as the values being imputed were actually observed rather than synthetically generated.

## Key features

The statistical matching imputer offers a truly non-parametric approach that operates without imposing restrictive assumptions about the underlying data distribution. This distribution-free nature makes it particularly valuable in scenarios where the data doesn't conform to common statistical assumptions or when the relationships are too complex to model parametrically.

One of the most compelling advantages of this method is its ability to preserve the empirical distribution of the imputed variables. Since the imputed values come directly from observed data points, the resulting dataset maintains the natural structure, variability, and relationships present in the original data. This preservation extends to features like multimodality, skewness, and natural bounds that might be lost in model-based approaches.

The technique demonstrates versatility in handling complex relationships between variables, particularly when there exists a good match across datasets. Without requiring explicit specification of interaction terms or functional forms, it naturally captures the intricate dependencies that exist in the data through the matching process. This makes it especially valuable for datasets where the relationships are not well understood or are difficult to express mathematically.

Perhaps most distinctively, the statistical matching approach returns actual observed values rather than modeled estimates. This characteristic ensures that the imputed values are realistic and plausible, as they represent real observations from similar data points. The method essentially says, "We have seen this pattern before, and here's what the missing values looked like in that situation," providing a grounded approach to filling in missing information.
