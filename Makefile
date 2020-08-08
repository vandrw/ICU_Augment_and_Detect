clean-results:
	@echo "Cleaning model saves and plots..."
	@rm -rvf categorization/model_saves/nose/*
	@rm -rvf categorization/model_saves/eyes/*
	@rm -rvf categorization/model_saves/skin/*
	@rm -rvf categorization/model_saves/mouth/*
	@rm -rvf categorization/model_saves/stacked/*
	@rm -rvf data/plots/*

clean-data:
	@echo "Removing data extracted from unparsed images..."
	@rm -rf data/parsed/validation_sick/*
	@rm -rf data/parsed/validation_healthy/*
	@rm -rf data/parsed/sick/*
	@rm -rf data/parsed/healthy/*
	@rm -rf data/parsed/brightened/sick/*
	@rm -rf data/parsed/brightened/healthy/*

create-data:
	@echo "Extracting data from images..."
	@python augment/face_org.py
	@python augment/alter_images.py