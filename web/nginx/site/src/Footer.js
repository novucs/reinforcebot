import React, {Component} from "react";
import {Container, Grid, Header, List, Segment} from "semantic-ui-react";


export default class Footer extends Component {
  render() {
    return (
      <Segment inverted vertical style={{padding: '5em 0em'}}>
        <Container>
          <Grid divided inverted stackable>
            <Grid.Row>
              <Grid.Column width={3}>
                <Header inverted as='h4' content='About'/>
                <List link inverted>
                  <List.Item as='a' href='start'>Quickstart</List.Item>
                  <List.Item as='a' href='mailto:william2.randall@live.uwe.ac.uk'>Contact Us</List.Item>
                </List>
              </Grid.Column>
              <Grid.Column width={7}>
                <Header as='h4' inverted>
                  ReinforceBot
                </Header>
                <p>
                  Automates the creation of software agents that can interact with the desktop environment.
                </p>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Container>
      </Segment>
    )
  }
}