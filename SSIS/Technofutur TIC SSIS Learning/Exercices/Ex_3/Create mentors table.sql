
USE X_Factor
GO

-- create a table to hold the mentors
CREATE TABLE tblMentor(

	-- the unique mentor number
	MentorId int PRIMARY KEY IDENTITY(1,1),

	-- the name of this mentor
	MentorName nvarchar(50)
)
-- now add the mentors one by one
INSERT tblMentor (MentorName) VALUES (N'Cheryl Cole')
INSERT tblMentor (MentorName) VALUES (N'Dannii Minogue')
INSERT tblMentor (MentorName) VALUES (N'Gary Barlow')
INSERT tblMentor (MentorName) VALUES (N'Kelly Rowland')
INSERT tblMentor (MentorName) VALUES (N'Louis Walsh')
INSERT tblMentor (MentorName) VALUES (N'Nicole Scherzinger')
INSERT tblMentor (MentorName) VALUES (N'Sharon Osbourne')
INSERT tblMentor (MentorName) VALUES (N'Simon Cowell')
INSERT tblMentor (MentorName) VALUES (N'Unknown')
INSERT tblMentor (MentorName) VALUES (N'David Wakefield')