USE Wistful
GO

CREATE TABLE Periods(
	Period nvarchar(255) NULL
)

GO
INSERT dbo.Periods (Period) VALUES (N'Cretaceous')
GO
INSERT dbo.Periods (Period) VALUES (N'Jurassic')
GO
INSERT dbo.Periods (Period) VALUES (N'Triassic')
GO

SELECT * FROM Periods