USE X_Factor
GO

-- create a table of food items
CREATE TABLE tblBushTuckerItem(
	BushTuckerItem int PRIMARY KEY,
	BushTuckerName nvarchar(255)
)

-- add in 4 tasty and nutritious items
INSERT INTO tblBushTuckerItem(BushTuckerItem,BushTuckerName) VALUES (1,'Locusts')
INSERT INTO tblBushTuckerItem(BushTuckerItem,BushTuckerName) VALUES (2,'Spiders')
INSERT INTO tblBushTuckerItem(BushTuckerItem,BushTuckerName) VALUES (3,'Centipedes')
INSERT INTO tblBushTuckerItem(BushTuckerItem,BushTuckerName) VALUES (4,'Worms')
	
-- show results
SELECT * FROM tblBushTuckerItem