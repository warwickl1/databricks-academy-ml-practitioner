# Core ML Issues
1. Keeping track of experiments or model development
2. Reproducing code
3. Comparing models
4. Standardisation of packaging and deploying models
**MLflow addresses these problems**

# MLflow Components
- **Tracking** - Record and query experiment: code, data, config, results
- **Projects** - Packaging format for reproducible runs on any platform
- **Models** - General model that supports diverse deployment tools
- **Model Registry** - Centralised and collaborative model lifecycle management
# The Full ML Lifecycle
MLflow does not replace the full ML lifecycle, but does improve upon it
1. Data prep & featurisation
	1. Data Scientists build features. Data Engineers provide infra for automating featurisation.
2. Model development
	1. Data Scientists build models and log them to MLflow, which records environment info
3. MLflow Tracking
	1. ^^
4. MLflow Model Registry
	1. Data Scientists move models to Staging
5. MLflow Deployment
	1. Deployment Engineers manage CI/CD tools which promote models to production
# MLflow Model Registry
1. Collaborative, centralised model hub
2. Facilitate experimentation, testing, and production
3. Integrate with approval and governance workflows
4. Monitor ML deployment and their performance